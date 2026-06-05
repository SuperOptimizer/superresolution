"""Prove the tiled streaming pipeline == processing the region whole (seam-free)."""
import sys, numpy as np
sys.path.insert(0,'/home/forrest/superresolution')
from superres import fysics_pipeline as fp

# synthetic volume with sheets + texture (in-RAM, acts as the 'zarr')
np.random.seed(0)
Z,Y,X = 200, 160, 160
vol = np.zeros((Z,Y,X), np.float32)
for z in range(Z):
    vol[z] = 0.3 + 0.4*((z//8)%2)              # layered sheets
vol += 0.3*np.sin(np.arange(X)*0.4)[None,None,:]  # in-plane texture
vol += np.random.randn(Z,Y,X)*0.05
vol = np.clip(vol,0,1); vol_u8 = (vol*255).astype(np.uint8)

md_phys = dict(delta_beta=1000, energy_kev=78, distance_mm=220, pixel_um=2.4,
               unsharp_sigma=1.2, unsharp_coeff=4.0)
# calibrate from a couple textured sub-cubes
samples = [vol[40:100,40:100,40:100]]
cal = fp.calibrate(md_phys, samples)
print(f"calib: db_scale={cal.db_scale:.2f} noise_ref={cal.noise_ref:.4f} eps={cal.guided_eps:.4f} halo={cal.halo}")

# (A) WHOLE: process the entire volume as one block (deconv+denoise, no diffusion)
whole = fp.process_tile(vol_u8, cal, do_deconv=True, do_denoise=True)

# (B) TILED: stream it through the pipeline with small tiles + halo
out = np.zeros_like(vol_u8)
def rd(z,y,x,dz,dy,dx): return vol_u8[z:z+dz,y:y+dy,x:x+dx]
def wr(z,y,x,blk): out[z:z+blk.shape[0],y:y+blk.shape[1],x:x+blk.shape[2]] = blk
stats = fp.run_pipeline(rd, wr, (Z,Y,X), cal, tile=64, occupancy_skip=False,
                        do_deconv=True, do_denoise=True)
print("tiles:", stats)

# compare in the INTERIOR (away from volume edges where halo is clamped/differs)
m = 24
a = whole[m:-m, m:-m, m:-m].astype(np.int16)
b = out[m:-m, m:-m, m:-m].astype(np.int16)
diff = np.abs(a-b)
print(f"interior tiled-vs-whole: max|d|={diff.max()} mean|d|={diff.mean():.3f} "
      f"(/255 -> {diff.mean()/255*100:.2f}% mean)")
print("SEAMLESS" if diff.max() <= 3 else f"SEAM MISMATCH (max {diff.max()})")
