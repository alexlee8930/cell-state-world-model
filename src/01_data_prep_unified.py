"""Unified Xenium data prep — runs ON Modal, downloads from 10x CDN directly (fast egress),
extracts 200k-cell patches + 422-gene expression + spatial coords + DINOv2-384 embeddings,
writes ONE npz to the mounted Volume /data (persists across jobs, zero re-upload)."""
import os, time, numpy as np
t0=time.time()
NAME="Xenium_V1_Human_Colon_Cancer_P1_CRC_Add_on_FFPE"
URL=f"https://cf.10xgenomics.com/samples/xenium/2.0.0/{NAME}/{NAME}_outs.zip"
os.makedirs("/tmp/x",exist_ok=True)

# 1. range-download only the members we need
from remotezip import RemoteZip
want=["cell_feature_matrix.h5","cells.csv.gz","morphology_focus/morphology_focus_0000.ome.tif"]
with RemoteZip(URL) as z:
    for m in want:
        z.extract(m,"/tmp/x"); print("got",m,round(time.time()-t0),"s",flush=True)

# 2. load morphology focus image (DAPI), full-res level
import tifffile
tf=tifffile.TiffFile("/tmp/x/morphology_focus/morphology_focus_0000.ome.tif")
# pyramidal: series[0].levels[0] is full res
plane=tf.series[0].levels[0].asarray()
if plane.ndim==3: plane=plane[0] if plane.shape[0]<plane.shape[-1] else plane[...,0]
print("plane",plane.shape,plane.dtype,round(time.time()-t0),"s",flush=True)
H,W=plane.shape

# 3. cells + expression
import pandas as pd, h5py, scipy.sparse as sp
cells=pd.read_csv("/tmp/x/cells.csv.gz")
with h5py.File("/tmp/x/cell_feature_matrix.h5","r") as f:
    g=f["matrix"]; data=g["data"][:]; indices=g["indices"][:]; indptr=g["indptr"][:]; shape=g["shape"][:]
    feats=[x.decode() for x in g["features"]["name"][:]]; ftype=[x.decode() for x in g["features"]["feature_type"][:]]
Xc=sp.csc_matrix((data,indices,indptr),shape=tuple(shape))  # genes x cells
gmask=np.array([t=="Gene Expression" for t in ftype])
Xg=Xc[gmask,:].tocsc(); genes=[feats[i] for i in range(len(feats)) if gmask[i]]
print("matrix",Xg.shape,"genes",len(genes),flush=True)

# 4. pixel coords + QC + patch-fit
px=0.2125; P=32
cx=(cells["x_centroid"].values/px).astype(int); cy=(cells["y_centroid"].values/px).astype(int)
tx=np.asarray(Xg.sum(0)).ravel()
qc=tx>=20; fit=(cx>=P)&(cx<W-P)&(cy>=P)&(cy<H-P)
keep=np.where(qc&fit)[0]; print("keep",len(keep),flush=True)
rng=np.random.default_rng(42); N=min(200000,len(keep))
sel=np.sort(rng.choice(keep,N,replace=False))

# 5. patches (uint8, percentile clip)
samp=rng.choice(sel,2000,replace=False)
vals=np.concatenate([np.asarray(plane[cy[i]-P:cy[i]+P,cx[i]-P:cx[i]+P]).ravel() for i in samp])
lo,hi=np.percentile(vals,[1,99.5]); print("clip",lo,hi,flush=True)
patches=np.zeros((N,64,64),np.uint8)
for j,i in enumerate(sel):
    p=np.asarray(plane[cy[i]-P:cy[i]+P,cx[i]-P:cx[i]+P],np.float32)
    patches[j]=(np.clip((p-lo)/(hi-lo),0,1)*255).astype(np.uint8)
print("patches",patches.shape,round(time.time()-t0),"s",flush=True)

# 6. expression normalize (CP-median + log1p + zscore)
Xsel=np.asarray(Xg[:,sel].todense()).T.astype(np.float32)
lib=Xsel.sum(1,keepdims=True); med=np.median(lib)
Xn=np.log1p(Xsel/lib*med); mu=Xn.mean(0); sd=Xn.std(0)+1e-8; Xz=(Xn-mu)/sd

# 7. DINOv2-384 embeddings (GPU)
import torch, torch.nn.functional as F
dev="cuda" if torch.cuda.is_available() else "cpu"
dino=torch.hub.load('facebookresearch/dinov2','dinov2_vits14').to(dev).eval()
mean=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1)
std=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)
Z=np.zeros((N,384),np.float32); BS=512
with torch.no_grad():
    for i in range(0,N,BS):
        b=torch.tensor(patches[i:i+BS].astype(np.float32)/255.0,device=dev).unsqueeze(1)
        b=F.interpolate(b,size=224,mode="bilinear",align_corners=False).repeat(1,3,1,1)
        b=(b-mean)/std
        Z[i:i+BS]=dino(b).cpu().numpy()
        if i%10240==0: print("dino",i,round(time.time()-t0),"s",flush=True)
print("Z",Z.shape,flush=True)

# 8. spatial coords (micron) + train/val split, save to Volume
xy=np.stack([cells["x_centroid"].values[sel],cells["y_centroid"].values[sel]],1).astype(np.float32)
perm=rng.permutation(N); val=perm[:8000]; tr=perm[8000:]
os.makedirs("/data",exist_ok=True)
np.savez("/data/xenium_unified.npz",
    expr=Xz.astype(np.float16), patches_u8=patches, dino=Z, xy=xy,
    tr_idx=tr, val_idx=val, all_genes=np.array(genes),
    norm_mu=mu, norm_sd=sd, norm_med=med, clip=np.array([lo,hi]))
os.makedirs("./out",exist_ok=True)
import json
json.dump(dict(N=int(N),genes=len(genes),plane=[int(H),int(W)],clip=[float(lo),float(hi)],
    wall_s=round(time.time()-t0)),open("./out/prep_summary.json","w"),indent=2)
print("DONE saved /data/xenium_unified.npz",round(time.time()-t0),"s",flush=True)
