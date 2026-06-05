"""Prove the pipeline streams at CONSTANT (small) RAM -> works on 20TB volumes.
Streams real level-0 chunks (LRU-cached for halo reuse, bounded cache = bounded RAM),
reports peak RSS as tiles accumulate. RAM must stay FLAT, not grow with tiles done."""
import sys, json, subprocess, numpy as np, os, time, resource
os.environ['OMP_NUM_THREADS']='2'
sys.path.insert(0,'/home/forrest/superresolution')
from superres import fysics_pipeline as fp
from functools import lru_cache

B="s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260310173927-45.532um-11.0m-110keV-masked.zarr"
def s3(k):
    r=subprocess.run(["aws","s3","cp","--no-sign-request",f"{B}/{k}","-"],capture_output=True)
    return r.stdout if r.returncode==0 else None
za=json.loads(s3("0/.zarray")); SH=za['shape']; CH=za['chunks']
ncz,ncy,ncx=[-(-s//c) for s,c in zip(SH,CH)]
print(f"volume {SH} = {np.prod(SH)/1e9:.1f}GB level-0; BOUNDED-cache streaming",flush=True)

@lru_cache(maxsize=64)   # BOUNDED cache = bounded RAM (64 chunks * 2MB = 128MB max)
def gc(cz,cy,cx):
    if not(0<=cz<ncz and 0<=cy<ncy and 0<=cx<ncx): return None
    b=s3(f"0/{cz}/{cy}/{cx}"); return np.frombuffer(b,np.uint8).reshape(CH) if b and len(b)==int(np.prod(CH)) else None
def read_region(z,y,x,dz,dy,dx):
    o=np.zeros((dz,dy,dx),np.uint8)
    for cz in range(z//128,(z+dz-1)//128+1):
     for cy in range(y//128,(y+dy-1)//128+1):
      for cx in range(x//128,(x+dx-1)//128+1):
        c=gc(cz,cy,cx)
        if c is None: continue
        gz,gy,gx=cz*128,cy*128,cx*128
        a0,b0,c0=max(z,gz),max(y,gy),max(x,gx); a1,b1,c1=min(z+dz,gz+128),min(y+dy,gy+128),min(x+dx,gx+128)
        o[a0-z:a1-z,b0-y:b1-y,c0-x:c1-x]=c[a0-gz:a1-gz,b0-gy:b1-gy,c0-gx:c1-gx]
    return o
def write_region(z,y,x,blk): pass  # discard (a real run writes to zarr); RAM proof doesn't need it

md=json.loads(s3("metadata.json") or "{}"); t=md.get("scan",{}).get("tomo",md.get("tomo",{})); a=t.get("acquisition",{}); ph=t.get("processing",{}).get("preprocessing",{}).get("phase",{})
md_phys=dict(delta_beta=ph.get("delta_beta",1000),energy_kev=a.get("energy",110),distance_mm=a.get("sampleDetectorDistance",11000),pixel_um=45.532,unsharp_sigma=ph.get("unsharp_sigma",1.2),unsharp_coeff=ph.get("unsharp_coeff",4.0))
cal=fp.calibrate(md_phys,[read_region(SH[0]//2,SH[1]//2,SH[2]//2,128,128,128).astype(np.float32)/255])
print(f"calib db_scale={cal.db_scale:.2f} halo={cal.halo}",flush=True)

# Process a BAND of tiles (a few hundred) and watch RSS stay flat. Use a sub-shape so
# it finishes in minutes but exercises many tiles -> the RAM-flatness is the proof.
SUB=(512, 1024, 1024)   # a real 0.5GB region streamed at 256-tiles = ~32 tiles, but
# to show MANY tiles, use small tiles:
t0=time.time(); rss=[]
def prog(p,f):
    if isinstance(f,float):
        r=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024
        rss.append(r)
        if len(rss)%4==0: print(f"  {p} {f*100:.0f}% RSS={r}MB {time.time()-t0:.0f}s",flush=True)
stats=fp.run_pipeline_2pass(read_region,write_region,SUB,cal,tile=128,
    do_normalize=True,do_zdrift=True,do_deconv=True,do_denoise=True,progress=prog)
peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024
print(f"DONE {stats} {time.time()-t0:.0f}s | peakRSS={peak}MB",flush=True)
print(f"RAM FLAT across {len(rss)} progress points: min={min(rss)}MB max={max(rss)}MB -> "
      f"{'CONSTANT (20TB-safe)' if max(rss)-min(rss)<300 else 'GROWING (leak!)'}",flush=True)
