#!/usr/bin/env python3
"""Read-only source-aware ONNX parity repair matrix for YOLO11s detector.

Use after auditing one approved source snapshot and the real weights, optionally
exporting one fresh candidate to an *absent* path. This tool never overwrites
Phase1, original .pt/.onnx, or prior results, nor claims automatic deployment.

The 0.001 raw-head project tolerance is immutable; a post-NMS agreement alone
cannot override its failure. Only real-model runs produce a measurement.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import platform
from pathlib import Path
import sys
import traceback

import numpy as np
from onnx_parity_diagnostics import sha256_file, verify_frozen_snapshot, verify_cohort_manifest, tensor_diagnostics
from onnx_repair_core import ParityGate, classify_variant_result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('repo','weights','baseline-onnx','image-root','cohort','baseline','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--candidate-output', type=Path,
                   help='New path to export a distinct, diagnostic ONNX candidate; original preserved')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--random-inputs', type=int, default=4)
    p.add_argument('--limit', type=int, default=0,
                   help='0 means evaluate every frozen image; smaller than 50 is forbidden')
    p.add_argument('--opt-levels', nargs='+', default=['DISABLE_ALL','ENABLE_ALL'],
                   choices=['DISABLE_ALL','ENABLE_BASIC','ENABLE_EXTENDED','ENABLE_ALL'])
    return p


@contextmanager
def torch_numeric_mode(torch, mkldnn_enabled):
    old = torch.backends.mkldnn.enabled
    try:
        torch.backends.mkldnn.enabled = mkldnn_enabled
        yield
    finally:
        torch.backends.mkldnn.enabled = old


def run_one(weights, onnx_path, images, seed, random_inputs, opt_level, mkldnn):
    import torch
    import onnxruntime as ort
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from PIL import Image
    from spv.instance_alignment import pair_detection_boxes

    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = YOLO(str(weights)).model.float().cpu().eval()
    options = ort.SessionOptions()
    options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, 'ORT_' + opt_level)
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=['CPUExecutionProvider'])
    xinfo = session.get_inputs()[0]
    if xinfo.type != 'tensor(float)' or list(xinfo.shape)[1:] != [3, 640, 640]:
        raise ValueError('ONNX input signature must be batch x 3 x 640 x 640 float32; observed '+str(xinfo.shape))
    if len(session.get_outputs()) != 1:
        raise ValueError('Exactly one raw detection output is required; model may have baked-in NMS')

    def compare(x):
        with torch_numeric_mode(torch, mkldnn), torch.no_grad():
            ref = model(x)
            if isinstance(ref, (tuple, list)):
                ref = ref[0]
            ref = ref.detach().float().cpu().numpy()
        candidate = session.run(None, {xinfo.name: x.numpy()})[0]
        if ref.shape[1] != 6 or ref.shape[0] != 1:
            raise ValueError('Expected YOLO11 cat/dog raw head (1, 6, N), got '+str(ref.shape))
        return tensor_diagnostics(ref, candidate)

    random_results = []
    for _ in range(random_inputs):
        x = torch.rand(1,3,640,640, dtype=torch.float32)
        random_results.append(compare(x))
    real_results=[]
    letterbox=LetterBox((640,640), auto=False)
    for index, path in enumerate(images):
        image = np.asarray(Image.open(path).convert('RGB'))
        lb=letterbox(image=image)
        x=torch.from_numpy(lb.transpose(2,0,1).copy()).float().unsqueeze(0)/255
        m=compare(x)
        real_results.append({'image_index':index,'sha256':sha256_file(path),
                             'overall':m['overall'],'box_max_abs':m['box_max_abs'],
                             'class_max_abs':m['class_max_abs'],'raw_passed':m['raw_passed'],
                             'largest_errors':m['largest_errors'][:3]})
    # Match each final detection exactly once, preserving class and confidence.
    # A raw-head pass alone does not establish stable post-NMS decisions.
    pt, ox = YOLO(str(weights)), YOLO(str(onnx_path), task='detect')
    same_count = same_cls = found = 0
    box_differences, confidence_differences, post_examples = [], [], []
    for index, path in enumerate(images):
        with torch_numeric_mode(torch, mkldnn):
            pa = pt.predict(str(path), conf=.40, iou=.50, imgsz=640,
                            rect=False, device='cpu', verbose=False)[0].boxes
        pb = ox.predict(str(path), conf=.40, iou=.50, imgsz=640,
                        rect=False, device='cpu', verbose=False)[0].boxes
        found += int(len(pa)>0)
        matched = (len(pa)==len(pb))
        class_equal = False
        bx_err = cf_err = None
        if matched:
            same_count+=1
            if len(pa):
                indices=pair_detection_boxes(pa.xyxy.cpu().numpy(), pb.xyxy.cpu().numpy())
                class_equal=bool((pa.cls.cpu().numpy()==pb.cls.cpu().numpy()[indices]).all())
                bx_err=float(np.abs(pa.xyxy.cpu().numpy()-pb.xyxy.cpu().numpy()[indices]).max())
                cf_err=float(np.abs(pa.conf.cpu().numpy()-pb.conf.cpu().numpy()[indices]).max())
                box_differences.append(bx_err)
                confidence_differences.append(cf_err)
            else:
                class_equal=True
            same_cls+=int(class_equal)
        if not matched or not class_equal or (bx_err is not None and bx_err>.001) or (cf_err is not None and cf_err>.001):
            post_examples.append({'image_index':index,'image_sha256':sha256_file(path),
                                  'pt_boxes':len(pa),'onnx_boxes':len(pb),
                                  'matched_class':class_equal,
                                  'max_box_difference_px':bx_err,
                                  'max_confidence_difference':cf_err})
    final={'images':len(images),'input':'square 640 RGB equivalent preprocessing',
           'same_number_of_detections':same_count/len(images),
           'same_classes':same_cls/len(images),
           'final_box_max_abs_diff_px':max(box_differences,default=0.),
           'final_confidence_max_abs_diff':max(confidence_differences,default=0.),
           'images_with_detections':found,'divergent_examples':post_examples}
    def agree_count(frac):
        count=round(frac*len(images))
        if abs(frac-count/len(images))>1e-8:raise ValueError('Inconsistent fraction denominator')
        return count
    gate=ParityGate(raw_random_max=max(q['overall']['max'] for q in random_results),
                    raw_real_max=max(q['overall']['max'] for q in real_results),
                    same_counts=agree_count(final['same_number_of_detections']),
                    same_classes=agree_count(final['same_classes']), images=len(images),
                    final_box_max_px=float(final['final_box_max_abs_diff_px']),
                    images_with_detections=found)
    decision=gate.check()
    # Explicitly require confidence parity in the final user-facing decision too.
    if decision['status']=='PASS' and final['final_confidence_max_abs_diff']>.001:
        decision.update(status='FAIL',final_passed=False,
                        reasons=['POST_NMS_CONFIDENCE_PARITY_FAILED'])
    return {'decision':decision,'variant':{'ort_optimization':opt_level,
                                          'torch_mkldnn':mkldnn,
                                          'onnx_sha256':sha256_file(onnx_path)},
            'random':{'n':len(random_results),'max_abs':gate.raw_random_max,
                      'largest_errors_by_input':[v['largest_errors'][:3] for v in random_results]},
            'real':{'n':len(real_results),'max_abs':gate.raw_real_max,
                    'box_max_abs':max(q['box_max_abs'] for q in real_results),
                    'class_max_abs':max(q['class_max_abs'] for q in real_results),
                    'per_image':real_results},
            'post_nms':final}


def main(argv=None):
    p=parser(); a=p.parse_args(argv)
    if a.output.exists(): p.error('Choose a NEW output path; prior evidence cannot be overwritten')
    if a.random_inputs<4 or a.limit and a.limit<50: p.error('Require >=4 random tensors and >=50 real images')
    if a.candidate_output and a.candidate_output.exists():p.error('Export candidate already exists; do not overwrite')
    for key in ('repo','weights','baseline_onnx','image_root','cohort','baseline'):
        if not getattr(a,key).exists():p.error(f'Missing {key}')
    out=a.output
    out.mkdir(parents=True,exist_ok=False)
    report={'status':'BLOCKED','phase1_modified':False,'deployment_allowed':False,
            'original_project_tolerance':0.001,
            'source':'approved-Drive-2026-10-30',
            'environment':{'python':platform.python_version()},'variants':[]}
    try:
        baseline=json.loads(a.baseline.read_text(encoding='utf-8'))
        src=verify_frozen_snapshot(a.repo,baseline['source_snapshot'])
        report['source']=src
        if not src['passed']: raise ValueError('Source drift: baseline cannot be trusted')
        weights=baseline['weights']
        for field,file,key in (('pt_sha',a.weights,'best.pt'),('baseline_onnx_sha',a.baseline_onnx,'best.onnx.bin')):
            sha=sha256_file(file)
            report[field]=sha
            if sha != weights[key]['sha256']: raise ValueError(f'{key} SHA mismatch')
        images=verify_cohort_manifest(a.image_root,a.cohort,min_images=50)
        if a.limit:images=images[:a.limit]
        report['cohort']={'count':len(images),'manifest_sha256':sha256_file(a.cohort)}
        sys.path.insert(0,str(a.repo.resolve()/'src'))
        import onnxruntime as ort
        import torch
        import ultralytics
        report['environment'].update(onnxruntime=ort.__version__,torch=torch.__version__,
                                     ultralytics=ultralytics.__version__)
        check=[('baseline',a.baseline_onnx)]
        if a.candidate_output:
            from spv.deploy import export_yolo_onnx
            candidate=export_yolo_onnx(a.weights,640,a.candidate_output)
            check.append(('new_export',candidate))
        for label,model in check:
            for level in a.opt_levels:
                for mkldnn in (True,False):
                    try:
                        rec=run_one(a.weights,model,images,a.seed,a.random_inputs,level,mkldnn)
                        rec['name']=f'{label}_{level}_mkldnn_{mkldnn}'
                        report['variants'].append(rec)
                    except Exception as e:
                        report['variants'].append({'name':f'{label}_{level}_mkldnn_{mkldnn}',
                                                   'decision':{'status':'BLOCKED'},
                                                   'error':str(e),'exception_type':type(e).__name__})
        success=[v for v in report['variants'] if classify_variant_result(v.get('decision',{}))
                 =='ELIGIBLE_FOR_INDEPENDENT_REVIEW']
        report['status']='PASS' if success else 'FAIL'
        report['eligible_variants']=[v['name'] for v in success]
        report['recommendation']=('Review candidate on separate regression, multiple-animal and T4 gates'
                                  if success else 'Keep PyTorch active; ONNX numerical parity not passed')
    except Exception as e:
        report['status']='BLOCKED'
        report['error']={'type':type(e).__name__,'detail':str(e)}
    finally:
        (out/'onnx_repair_matrix.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print(f"{report['status']}: {out/'onnx_repair_matrix.json'}")
    return 0 if report['status']=='PASS' else 2

if __name__=='__main__':raise SystemExit(main())
