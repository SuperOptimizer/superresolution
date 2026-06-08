import os,sys,ctypes as C,numpy as np
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superres import s3zarr, fysics_pipeline as fp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
f32p=C.POINTER(C.c_float); L=fp.lib()
L.fy_musica2d.argtypes=[f32p,f32p,C.c_int,C.c_int,C.c_int,C.c_float,C.c_float]; L.fy_musica2d.restype=C.c_int
v6=s6=s3zarr.open_local(os.path.expanduser("~/paris4_2um_processed_v6"),0)
# full 1024^2 Z mid-plane
sl=v6.read_region(512,0,0,1,1024,1024)[0].astype(np.float32)/255
def nrm(a):
    v=a[a>0]; lo,hi=(np.percentile(v,1),np.percentile(v,99)) if v.size else (0,1)
    return np.clip((a-lo)/(hi-lo+1e-9),0,1)
base=nrm(sl)
def mus(a,p,core=0.0,lv=4):
    o=np.empty_like(a);ny,nx=a.shape
    L.fy_musica2d(np.ascontiguousarray(a).ctypes.data_as(f32p),o.ctypes.data_as(f32p),ny,nx,lv,C.c_float(p),C.c_float(core));return np.clip(o,0,1)
panels=[("v6 signal-recovery",base),("MUSICA p=0.8 (gentle)",mus(base,0.8)),
        ("MUSICA p=0.6 (medium)",mus(base,0.6)),("MUSICA p=0.5 (strong)",mus(base,0.5))]
fig,ax=plt.subplots(1,4,figsize=(30,8))
for i,(t,im) in enumerate(panels):
    ax[i].imshow(im,cmap="gray");ax[i].set_title(t,fontsize=14);ax[i].axis("off")
fig.tight_layout();fig.savefig(os.path.expanduser("~/v6_musica.png"),dpi=110,bbox_inches="tight");plt.close(fig)
# also a 2-panel: v6 vs best MUSICA, full res
for nm,im in [("v6_musica_before",base),("v6_musica_after",mus(base,0.6))]:
    from PIL import Image; Image.fromarray((im*255).astype(np.uint8)).save(os.path.expanduser(f"~/{nm}.png"))
print("wrote ~/v6_musica.png + before/after")
