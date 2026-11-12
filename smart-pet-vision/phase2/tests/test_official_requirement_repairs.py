"""Exercise clean notebook discovery and real training provenance boundaries without training."""
import ast
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def resolver_from_notebook(name):
    document = json.loads((ROOT/'notebooks'/name).read_text())
    source = ''.join(document['cells'][1]['source'])
    start = source.index('def _resolve_spv_repo(')
    end = source.index('\nif IN_COLAB:', start)
    env = {'Path': Path}
    exec(compile(ast.parse(source[start:end]), name, 'exec'), env)
    return env['_resolve_spv_repo']


class Phase2NotebookDiscovery(unittest.TestCase):
    def test_parent_and_phase2_paths_select_phase2_not_phase1(self):
        for name in ['01_data_preparation.ipynb', '02_classification.ipynb',
                     '03_detection.ipynb', '04_segmentation.ipynb', '05_integration.ipynb']:
            with self.subTest(notebook=name):
                resolve = resolver_from_notebook(name)
                self.assertEqual(resolve(ROOT.parent), ROOT)
                self.assertEqual(resolve(ROOT, str(ROOT.parent)), ROOT)
                self.assertEqual(resolve(ROOT.parent.parent, str(ROOT.parent.parent)), ROOT)

    def test_cached_phase1_module_requires_runtime_restart(self):
        d=json.loads((ROOT/'notebooks/03_detection.ipynb').read_text())
        source=''.join(d['cells'][1]['source'])
        start=source.index('for _module_name in (')
        end=source.index('from spv.common', start)
        env={'Path':Path,'REPO':ROOT,'sys':types.SimpleNamespace(modules={'spv.common':types.SimpleNamespace(__file__=str(ROOT.parent/'src/spv/common.py'))})}
        with self.assertRaisesRegex(RuntimeError, 'Restart'):
            exec(compile(source[start:end], '<bootstrap>', 'exec'), env)
        env['sys'].modules={'spv.common':types.SimpleNamespace(__file__=str(ROOT/'src/spv/common.py'))}
        exec(compile(source[start:end], '<bootstrap>', 'exec'), env)

    def test_missing_or_ambiguous_sources_are_rejected(self):
        resolve = resolver_from_notebook('03_detection.ipynb')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with self.assertRaises(FileNotFoundError): resolve(root)
            for name in ['a/smart-pet-vision/phase2','b/smart-pet-vision/phase2']:
                p=root/name
                (p/'src/spv').mkdir(parents=True);(p/'configs').mkdir()
                for n in ['src/spv/detect.py','src/spv/segment.py','configs/config.yaml']:(p/n).write_text('')
            with self.assertRaises(FileNotFoundError): resolve(root)


class TrainingIdentity(unittest.TestCase):
    def setUp(self):
        from spv import training_provenance
        self.module=training_provenance

    def test_identity_persists_and_updates_only_an_existing_wandb_run(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);yaml=root/'data.yaml';yaml.write_text('names: [cat, dog]\n')
            callbacks={};model=types.SimpleNamespace(add_callback=lambda name,fn:callbacks.setdefault(name,fn))
            updates=[];wandb=types.SimpleNamespace(run=types.SimpleNamespace(config=types.SimpleNamespace(update=lambda *a,**kw:updates.append((a,kw)))))
            identity={'commit':'a'*40,'source_fingerprint':{'sha256':'b'*64}}
            with patch.object(self.module,'record',return_value=identity),patch.dict(sys.modules,{'wandb':wandb}):
                result=self.module.attach_training_provenance(model,{'paths':{'runs':root},'seed':42},yaml,'new_run','detect')
                callbacks['on_train_start'](None)
            self.assertEqual(json.loads((root/'detect/new_run/source_provenance.json').read_text()),identity)
            self.assertEqual(updates[0][0][0]['source_commit'],'a'*40)
            self.assertFalse(updates[0][1]['allow_val_change'])

    def test_existing_training_outputs_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);yaml=root/'data.yaml';yaml.write_text('x');old=root/'detect/run';old.mkdir(parents=True);(old/'best.pt').write_bytes(b'historical')
            model=types.SimpleNamespace(add_callback=lambda *a:None)
            with patch.object(self.module,'record',return_value={}):
                with self.assertRaises(FileExistsError):self.module.attach_training_provenance(model,{'paths':{'runs':root},'seed':42},yaml,'run','detect')
            self.assertEqual((old/'best.pt').read_bytes(),b'historical')

    def test_absent_wandb_run_does_not_create_a_remote_run(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);yaml=root/'data.yaml';yaml.write_text('x');callbacks={}
            model=types.SimpleNamespace(add_callback=lambda key,fn:callbacks.setdefault(key,fn))
            with patch.object(self.module,'record',return_value={}),patch.dict(sys.modules,{'wandb':types.SimpleNamespace(run=None)}):
                self.module.attach_training_provenance(model,{'paths':{'runs':root},'seed':42},yaml,'run','segment')
                callbacks['on_train_start'](None)

if __name__=='__main__':unittest.main()
