"""Embed Xenium morphology patches with pretrained DINOv2 (ViT-S/14).
Grayscale 64x64 -> resize 224 -> 3ch -> DINOv2 CLS embedding (384-d).
Outputs ./out/morphology_embeddings.npz
"""
import os, numpy as np, torch, torch.nn.functional as F, time
os.makedirs("./out", exist_ok=True)
dev="cuda" if torch.cuda.is_available() else "cpu"
d=np.load("xenium_pairs_full.npz", allow_pickle=True)
patches=d["patches_u8"].astype(np.float32)/255.0   # (N,64,64)
expr=d["expr"]; tr=d["tr_idx"]; va=d["val_idx"]
N=len(patches); print(f"device={dev} N={N}", flush=True)

# load DINOv2 ViT-S/14 via torch.hub (downloads weights)
model=torch.hub.load('facebookresearch/dinov2','dinov2_vits14').to(dev).eval()
print("DINOv2 vits14 loaded", flush=True)

# ImageNet normalization
mean=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1)
std=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)

def embed(idx, bs=256):
    outs=[]
    for i in range(0,len(idx),bs):
        b=patches[idx[i:i+bs]]
        x=torch.tensor(b,device=dev).unsqueeze(1)              # (b,1,64,64)
        x=F.interpolate(x,size=(224,224),mode="bilinear",align_corners=False)
        x=x.repeat(1,3,1,1)                                    # 3ch
        x=(x-mean)/std
        with torch.no_grad():
            emb=model(x)                                       # (b,384) CLS
        outs.append(emb.cpu().numpy())
    return np.concatenate(outs,0)

t0=time.time()
all_idx=np.arange(N)
Z=embed(all_idx)
print(f"embedded {Z.shape} in {time.time()-t0:.0f}s", flush=True)
np.savez("./out/morphology_embeddings.npz",
    Z=Z.astype(np.float32), expr=expr.astype(np.float32),
    tr_idx=tr, val_idx=va, all_genes=d["all_genes"], sel=d["sel"])
print("saved morphology_embeddings.npz", flush=True)
