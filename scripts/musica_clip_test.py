import os,sys,ctypes as C,numpy as np
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superres import s3zarr, fysics_pipeline as fp
f32p=C.POINTER(C.c_float); L=fp.lib()
L.fy_musica2d.argtypes=[f32p,f32p,C.c_int,C.c_int,C.c_int,C.c_float,C.c_float]; L.fy_musica2d.restype=C.c_int
# use the RAW cached volume (has the export clip / saturation), a slice with saturated pixels
raw=s3zarr.open_local(os.path.expanduser("~/paris4_2um_v3_raw"),0)
sl=raw.read_region(512,0,0,1,1024,1024)[0].astype(np.float32)/255
u8=(sl*255).astype(np.uint8)
print(f"slice saturation: u8==255: {(u8==255).sum()}  u8>=254: {(u8>=254).sum()}  u8==0: {(u8==0).sum()}  ({(u8==255).mean()*100:.3f}% at 255)")
# tail of histogram -> the spike
h=np.bincount(u8.ravel(),minlength=256)
print(f"hist tail h[250..255]: {list(h[250:256])}  (spike at 255 = clip)")
def mus(a):
    o=np.empty_like(a);ny,nx=a.shape
    L.fy_musica2d(np.ascontiguousarray(a).ctypes.data_as(f32p),o.ctypes.data_as(f32p),ny,nx,4,C.c_float(0.6),C.c_float(0.0));return np.clip(o,0,1)
# A: naive MUSICA on the whole slice
A=mus(sl)
# B: rails-excluded -- replace 0/255 with the local median of valid neighbors before MUSICA,
#    run MUSICA, then restore the rails as-is in the output
from scipy.ndimage import median_filter
rail=(u8>=254)|(u8==0)
filled=sl.copy()
if rail.any():
    med=median_filter(sl,size=5)
    filled[rail]=med[rail]   # inpaint rails with local median so they don't create false edges
B=mus(filled); B[rail]=sl[rail]  # leave rails as-is in the output
# compare: how different is the enhancement NEAR saturated pixels (the contamination zone)?
from scipy.ndimage import binary_dilation
near=binary_dilation(rail,iterations=8)&~rail
far=~binary_dilation(rail,iterations=12)
d=np.abs(A-B)
print(f"\nMUSICA naive vs rails-handled: mean|diff| overall={d.mean()*255:.2f} u8")
print(f"  NEAR saturated pixels (8-vox ring): {d[near].mean()*255:.2f} u8")
print(f"  FAR from saturation:                {d[far].mean()*255:.2f} u8")
print(f"  -> if NEAR >> FAR, the clip DOES contaminate MUSICA near rails")
# also: does naive MUSICA blow up the saturation region? count new 255s it creates near rails
print(f"\nnaive MUSICA pushed-to-255 NEAR rails: {(A[near]>=0.99).mean()*100:.1f}%  FAR: {(A[far]>=0.99).mean()*100:.2f}%")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def nrm(a): v=a[a>0];lo,hi=(np.percentile(v,1),np.percentile(v,99));return np.clip((a-lo)/(hi-lo+1e-9),0,1)
fig,ax=plt.subplots(1,3,figsize=(24,8))
ax[0].imshow(nrm(sl),cmap="gray");ax[0].set_title("raw (with clip)");ax[0].axis("off")
ax[1].imshow(A,cmap="gray");ax[1].set_title("MUSICA naive");ax[1].axis("off")
ax[2].imshow(B,cmap="gray");ax[2].set_title("MUSICA rails-excluded");ax[2].axis("off")
fig.tight_layout();fig.savefig(os.path.expanduser("~/musica_clip.png"),dpi=100,bbox_inches="tight")
print("wrote ~/musica_clip.png")
