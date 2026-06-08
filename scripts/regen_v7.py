import os,sys,numpy as np
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superres import s3zarr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
raw=s3zarr.open_local(os.path.expanduser("~/paris4_2um_v3_raw"),0)
v6=s3zarr.open_local(os.path.expanduser("~/paris4_2um_processed_v6"),0)
v7=s3zarr.open_local(os.path.expanduser("~/paris4_2um_processed_v7"),0)
H=os.path.expanduser("~"); N=1024; mid=N//2
def nrm(a):
    v=a[a>0]; lo,hi=(np.percentile(v,1),np.percentile(v,99)) if v.size else (0,1)
    return np.clip((a-lo)/(hi-lo+1e-9),0,1)
for ax in ("Z","Y","X"):
    if ax=="Z": b=raw.read_region(mid,0,0,1,N,N)[0]; s6=v6.read_region(mid,0,0,1,N,N)[0]; s7=v7.read_region(mid,0,0,1,N,N)[0]
    elif ax=="Y": b=raw.read_region(0,mid,0,N,1,N)[:,0,:]; s6=v6.read_region(0,mid,0,N,1,N)[:,0,:]; s7=v7.read_region(0,mid,0,N,1,N)[:,0,:]
    else: b=raw.read_region(0,0,mid,N,N,1)[:,:,0]; s6=v6.read_region(0,0,mid,N,N,1)[:,:,0]; s7=v7.read_region(0,0,mid,N,N,1)[:,:,0]
    fig,axx=plt.subplots(1,3,figsize=(30,10))
    for i,(t,im) in enumerate([("raw",b),("v6 signal-recovery",s6),("v7 +MUSICA",s7)]):
        axx[i].imshow(nrm(im.astype(np.float32)),cmap="gray");axx[i].set_title(f"{ax} {t}",fontsize=14);axx[i].axis("off")
    fig.tight_layout();fig.savefig(os.path.join(H,f"v7_{ax}_compare.png"),dpi=105,bbox_inches="tight");plt.close(fig)
    print(f"wrote v7_{ax}_compare.png",flush=True)
print("done")
