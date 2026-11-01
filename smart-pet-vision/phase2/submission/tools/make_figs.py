import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
S=str(Path(__file__).resolve().parents[1])
EV=str(Path(__file__).resolve().parents[2]/'report'/'evidence')
H=json.load(open(S+'/tools/wandb_history.json'))
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.25})
BL,OR,GR="#2f6db5","#d9822b","#3a9a5b"
def curves(run,lt,lv,m1,m2,title,out,m1lab,m2lab):
    h=H[run]; ep=[r["epoch"] for r in h]
    fig,ax=plt.subplots(1,2,figsize=(9,3.1))
    ax[0].plot(ep,[r[lt] for r in h],color=BL,label="train"); ax[0].plot(ep,[r[lv] for r in h],color=OR,label="validation")
    ax[0].axvline(30.5,ls="--",color="grey",lw=.8); ax[0].text(31,ax[0].get_ylim()[1]*.97,"mosaic off",fontsize=8,color="grey",va="top")
    ax[0].set_title(title+" loss"); ax[0].set_xlabel("Epoch"); ax[0].legend(frameon=False)
    ax[1].plot(ep,[r[m1] for r in h],color=GR,label=m1lab); ax[1].plot(ep,[r[m2] for r in h],color=BL,label=m2lab)
    best=max(h,key=lambda r:0.1*r[m1]+0.9*r[m2])
    ax[1].scatter([best["epoch"]],[best[m2]],color=BL,zorder=3); ax[1].annotate(f'kept: epoch {best["epoch"]}, {best[m2]:.3f}',(best["epoch"],best[m2]),xytext=(-120,-30),textcoords="offset points",fontsize=8,arrowprops=dict(arrowstyle="-",color="grey"))
    ax[1].set_ylim(0,1.02); ax[1].set_title("Validation metric"); ax[1].set_xlabel("Epoch"); ax[1].legend(frameon=False,loc="lower right")
    fig.tight_layout(); fig.savefig(out,dpi=200); plt.close(fig); return best
b=curves("2seovilv","train/box_loss","val/box_loss","metrics/mAP50(B)","metrics/mAP50-95(B)","Box",S+"/figures/M4_training_curves.png","mAP50","mAP50-95")
print("det best",b["epoch"],b["metrics/mAP50(B)"],b["metrics/mAP50-95(B)"])
b=curves("3ztpz2tc","train/seg_loss","val/seg_loss","metrics/mAP50(M)","metrics/mAP50-95(M)","Mask",S+"/figures/M5_training_curves.png","mask mAP50","mask mAP50-95")
print("seg best",b["epoch"],b["metrics/mAP50(M)"],b["metrics/mAP50-95(M)"])
# benchmark figure: p50/p95/p99 per measurement
pt=json.load(open(EV+"/M8_benchmark_pytorch.json")); ox=json.load(open(EV+"/M8_benchmark_onnx_detector.json"))
th=json.load(open(EV+"/M8_service_http.json")); ch=json.load(open(EV+"/P2_cpu_http_validation_mu2tolkg.json"))
rows=[("PyTorch, T4\nin-process",pt["latency_ms"]),("ONNX detector (CPU)\n+ T4, in-process",ox["latency_ms"]),("HTTP, T4\n1 worker",th["http_latency_ms"]),("HTTP, CPU only\n1 worker",ch["http_latency_ms"])]
import numpy as np
fig,ax=plt.subplots(figsize=(8.2,3.3)); x=np.arange(len(rows)); w=.26
for i,(k,c) in enumerate([("p50",BL),("p95",OR),("p99","#8c5fb0")]):
    v=[r[1][k] for r in rows]; ax.bar(x+(i-1)*w,v,w,color=c,label=k)
    for xi,vi in zip(x+(i-1)*w,v): ax.text(xi,vi+8,f"{vi:.0f}",ha="center",fontsize=7.5)
ax.axhline(2000,color="grey")
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows],fontsize=8.5); ax.set_ylabel("Latency per image (ms)"); ax.legend(frameon=False,ncol=3,loc="upper left")
ax.set_ylim(0,780); ax.text(3.45,760,"A8 limit 2,000 ms (off scale)",ha="right",va="top",fontsize=8,color="grey")
fig.tight_layout(); fig.savefig(S+"/figures/M8_latency_by_backend.png",dpi=200); plt.close(fig)
# service figure: rendered from saved responses
fig=plt.figure(figsize=(8.4,3.6)); fig.patch.set_facecolor("white")
txt=("$ uvicorn spv.serve:app --host 127.0.0.1 --port 8765 --workers 1\n"
"$ curl http://127.0.0.1:8765/health\n"
'{"status": "ok", "detector": "best.pt", "segmenter": "best.pt", "classifier": "finetune"}\n\n'
'$ curl -X POST -F "file=@photo.jpg" "http://127.0.0.1:8765/analyze?images=false"   # x300\n'
f'HTTP 200 x {th["status_codes"]["200"]}   p50/p95/p99 = {th["http_latency_ms"]["p50"]:.0f}/{th["http_latency_ms"]["p95"]:.0f}/{th["http_latency_ms"]["p99"]:.0f} ms   (T4)\n'
'every response: "staff_approval_required": true, "automatic_publication_allowed": false\n\n'
'$ curl -X POST -F "file=@x.txt;type=text/plain" .../analyze     ->  HTTP 415\n'
'$ curl -X POST -F "file=@empty.jpg" .../analyze                  ->  HTTP 400')
fig.text(0.02,0.96,txt,family="monospace",fontsize=8.2,va="top")
fig.savefig(S+"/figures/M8_service_session.png",dpi=200,bbox_inches="tight",pad_inches=0.15); plt.close(fig)
