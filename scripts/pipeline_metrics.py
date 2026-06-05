import sys, json, subprocess, numpy as np
sys.path.insert(0,'/home/forrest/superresolution')
from superres import fysics_pipeline as fp
from numpy.fft import fftn, fftshift
from scipy.ndimage import gaussian_filter

BUCKET="vesuvius-challenge-open-data"; SCROLL="PHerc0139"
ZARR="volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr"
def s3(k):
    r=subprocess.run(["aws","s3","cp","--no-sign-request",f"s3://{BUCKET}/{SCROLL}/{ZARR}/{k}","-"],capture_output=True)
    return r.stdout if r.returncode==0 else None
def chunk(z,y,x):
    b=s3(f"0/{z}/{y}/{x}"); return np.frombuffer(b,np.uint8).reshape(128,128,128) if b and len(b)==128**3 else np.zeros((128,128,128),np.uint8)
md=json.loads(s3("metadata.json")); t=md.get("scan",{}).get("tomo",md.get("tomo",{})); acq=t.get("acquisition",{})
ph=t.get("processing",{}).get("preprocessing",{}).get("phase",{})
md_phys=dict(delta_beta=ph.get("delta_beta",1000),energy_kev=acq.get("energy",78),distance_mm=acq.get("sampleDetectorDistance",220),pixel_um=acq.get("detector",{}).get("samplePixelSize",0.0024)*1000,unsharp_sigma=ph.get("unsharp_sigma",1.2),unsharp_coeff=ph.get("unsharp_coeff",4.0))
cz,cy,cx=300,103,103; vol=np.zeros((256,256,256),np.uint8)
for dz in(0,1):
 for dy in(0,1):
  for dx in(0,1): vol[dz*128:dz*128+128,dy*128:dy*128+128,dx*128:dx*128+128]=chunk(cz+dz,cy+dy,cx+dx)
cal=fp.calibrate(md_phys,[vol[64:192,64:192,64:192].astype(np.float32)/255])
out=np.zeros_like(vol)
def rd(z,y,x,dz,dy,dx): return vol[z:z+dz,y:y+dy,x:x+dx]
def wr(z,y,x,b): out[z:z+b.shape[0],y:y+b.shape[1],x:x+b.shape[2]]=b
fp.run_pipeline(rd,wr,(256,256,256),cal,tile=128,do_deconv=True,do_denoise=True,do_diffusion=False)

# ---- metrics (interior, avoid edges) ----
A=vol[32:224,32:224,32:224].astype(np.float32); B=out[32:224,32:224,32:224].astype(np.float32)
def radial_psd(v):
    v=v-v.mean(); F=np.abs(fftshift(fftn(v)))**2; n=v.shape[0]; c=n//2
    z,y,x=np.indices(v.shape)-c; r=np.sqrt(z*z+y*y+x*x).astype(int)
    return (np.bincount(r.ravel(),F.ravel())/np.maximum(np.bincount(r.ravel()),1))[:c]
pA,pB=radial_psd(A),radial_psd(B); nyq=len(pA)-1
def band(p,a,b): return p[int(a*nyq):int(b*nyq)].sum()
def flatnoise(v): 
    r=v-gaussian_filter(v,1.5); 
    # low-gradient blocks
    sub=[]; 
    for z in range(0,v.shape[0]-16,16):
     for y in range(0,v.shape[1]-16,16):
      for x in range(0,v.shape[2]-16,16):
        p=v[z:z+16,y:y+16,x:x+16]
        if (p>15).mean()>0.9 and p.std()<25: sub.append((p-gaussian_filter(p,1.5)).std())
    return np.median(sub) if sub else (v-gaussian_filter(v,1.5)).std()
def texcontrast(v):
    return float(np.std(v-gaussian_filter(v,1.0)))
m={
 "finite": bool(np.isfinite(B).all()),
 "occupancy_in": float((A>20).mean()), "occupancy_out": float((B>20).mean()),
 "clip_lo_out": float((out==0).mean()), "clip_hi_out": float((out==255).mean()),
 "midband_gain": band(pB,0.17,0.5)/max(band(pA,0.17,0.5),1e-9),
 "highband_gain": band(pB,0.5,1.0)/max(band(pA,0.5,1.0),1e-9),
 "flatnoise_in": float(flatnoise(A)), "flatnoise_out": float(flatnoise(B)),
 "texcontrast_in": texcontrast(A), "texcontrast_out": texcontrast(B),
}
m["noise_ratio"]= m["flatnoise_out"]/max(m["flatnoise_in"],1e-9)
m["tex_ratio"]= m["texcontrast_out"]/max(m["texcontrast_in"],1e-9)
m["tex_over_noise_gain"]= m["tex_ratio"]/max(m["noise_ratio"],1e-9)
json.dump(m, open("/tmp/pmetrics.json","w"), indent=1)
for k,v in m.items(): print(f"{k:20s} {v}")
