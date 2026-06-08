import os,sys,numpy as np
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superres import s3zarr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
proc=s3zarr.open_local(os.path.expanduser("~/paris4_2um_processed_v3"),0)
raw=s3zarr.open_local(os.path.expanduser("~/paris4_2um_v3_raw"),0)
H=os.path.expanduser("~"); N=1024; mid=N//2
def nrm(a):
    v=a[a>0]; 
    if v.size==0: return a
    lo,hi=np.percentile(v,1),np.percentile(v,99); return np.clip((a-lo)/(hi-lo+1e-9),0,1)
for ax in ("Z","Y","X"):
    if ax=="Z": b=raw.read_region(mid,0,0,1,N,N)[0]; a=proc.read_region(mid,0,0,1,N,N)[0]
    elif ax=="Y": b=raw.read_region(0,mid,0,N,1,N)[:,0,:]; a=proc.read_region(0,mid,0,N,1,N)[:,0,:]
    else: b=raw.read_region(0,0,mid,N,N,1)[:,:,0]; a=proc.read_region(0,0,mid,N,N,1)[:,:,0]
    nb,na=nrm(b.astype(np.float32)),nrm(a.astype(np.float32))
    fig,axx=plt.subplots(1,2,figsize=(22,11))
    axx[0].imshow(nb,cmap="gray");axx[0].set_title(f"{ax} BEFORE");axx[0].axis("off")
    axx[1].imshow(na,cmap="gray");axx[1].set_title(f"{ax} AFTER v3 (global+local air cut)");axx[1].axis("off")
    fig.tight_layout();fig.savefig(os.path.join(H,f"optv3_{ax}_compare.png"),dpi=110,bbox_inches="tight");plt.close(fig)
    print(f"wrote optv3_{ax}_compare.png",flush=True)
print("done")
