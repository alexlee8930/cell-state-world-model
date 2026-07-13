"""
World Model v2 — GPU training on Modal (A100-40GB).
Full-scale version of the CPU MVP:
  - 256x256 native resolution (CPU MVP downsampled to 128)
  - VQ-VAE tokenizer (CPU MVP used continuous VAE)
  - Transformer dynamics over latent tokens + Genie-style latent action
  - longer training, cosine schedule
Input : wm_sequences.npy  (54, 40, 256, 256) float32, per-seq normalized
Output: everything under ./out/  (harvested by Modal)
"""
import os, json, time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

OUT = "out"; os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, torch.cuda.get_device_name() if dev=="cuda" else "")

# ---------- data: use Volume cache, else regenerate from IDR S3 ----------
def ensure_data():
    cache = "/data/wm_sequences.npy"
    local = "wm_sequences.npy"
    if os.path.exists(local):
        print("using local", local); return local
    if os.path.exists(cache):
        print("using cached", cache); return cache
    print("regenerating from IDR S3 ...", flush=True)
    import io, time, urllib.request, numcodecs
    from concurrent.futures import ThreadPoolExecutor, as_completed
    BASE="https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0052A/5514375.zarr/0"
    Tn,Cn,Zn=40,3,31; comp=numcodecs.Blosc()
    def fc(t,c,z,retries=4):
        url=f"{BASE}/{t}/{c}/{z}/0/0"
        for a in range(retries):
            try:
                with urllib.request.urlopen(url,timeout=60) as r: raw=r.read()
                return np.frombuffer(comp.decode(raw),dtype="<u2").reshape(256,256)
            except Exception:
                if a==retries-1: raise
                time.sleep(1.5*(a+1))
    vol=np.zeros((Tn,Cn,Zn,256,256),dtype=np.uint16)
    jobs=[(t,c,z) for t in range(Tn) for c in range(Cn) for z in range(Zn)]
    t0=time.time(); done=0
    with ThreadPoolExecutor(max_workers=16) as ex:
        fut={ex.submit(fc,t,c,z):(t,c,z) for t,c,z in jobs}
        for f in as_completed(fut):
            t,c,z=fut[f]; vol[t,c,z]=f.result(); done+=1
            if done%600==0: print(f"  {done}/{len(jobs)} ({round(time.time()-t0)}s)",flush=True)
    dna=vol[:,1].mean(axis=(0,2,3)); thr=np.percentile(dna,40)
    zk=[z for z in range(Zn) if dna[z]>thr]
    seqs=[]
    for c in range(Cn):
        for z in zk:
            s=vol[:,c,z].astype(np.float32); lo,hi=np.percentile(s,1),np.percentile(s,99.5)
            seqs.append(np.clip((s-lo)/(hi-lo+1e-6),0,1))
    seqs=np.stack(seqs); print("regenerated:",seqs.shape,f"{round(time.time()-t0)}s")
    if os.path.isdir("/data"):
        np.save(cache,seqs); print("cached to",cache)
        return cache
    np.save(local,seqs); return local

DATA_PATH = ensure_data()

# LPIPS perceptual loss — used when available (installed in the GPU image);
# falls back to None on the local CPU smoke env so the script still runs.
try:
    import lpips as _lpips_mod
    LPIPS = _lpips_mod.LPIPS(net="vgg").to(dev)
    for p in LPIPS.parameters(): p.requires_grad_(False)
    print("LPIPS: enabled (vgg)")
except Exception as _e:
    LPIPS = None
    print("LPIPS: unavailable ->", type(_e).__name__)
LPIPS_W = 0.1

seqs = np.load(DATA_PATH)                      # (54,40,256,256)
S, T, H, W = seqs.shape
print("data:", seqs.shape)
X = torch.from_numpy(seqs).float()            # already 0..1 normalized
TRAIN_T = 32                                  # hold out last 8 frames

# ---------- VQ-VAE tokenizer ----------
class VQ(nn.Module):
    def __init__(self, K=512, D=64):
        super().__init__()
        self.K, self.D = K, D
        self.emb = nn.Embedding(K, D); self.emb.weight.data.uniform_(-1/K, 1/K)
    def forward(self, z):                      # z: (B,D,h,w)
        B,D,h,w = z.shape
        zf = z.permute(0,2,3,1).reshape(-1, D)
        d = (zf**2).sum(1,keepdim=True) - 2*zf@self.emb.weight.t() + (self.emb.weight**2).sum(1)
        idx = d.argmin(1)
        zq = self.emb(idx).view(B,h,w,D).permute(0,3,1,2)
        loss = F.mse_loss(zq.detach(), z) + 0.25*F.mse_loss(zq, z.detach())
        zq = z + (zq - z).detach()             # straight-through
        return zq, loss, idx.view(B,h,w)

def conv(i,o,s=2): return nn.Sequential(nn.Conv2d(i,o,4,s,1), nn.GroupNorm(8,o), nn.SiLU())
def up(i,o):       return nn.Sequential(nn.ConvTranspose2d(i,o,4,2,1), nn.GroupNorm(8,o), nn.SiLU())

class Enc(nn.Module):   # 256 -> 16  (D channels)
    def __init__(s,D=64):
        super().__init__()
        s.net = nn.Sequential(conv(1,32),conv(32,64),conv(64,128),conv(128,128), nn.Conv2d(128,D,3,1,1))
    def forward(s,x): return s.net(x)
class Dec(nn.Module):   # 16 -> 256
    def __init__(s,D=64):
        super().__init__()
        s.net = nn.Sequential(up(D,128),up(128,128),up(128,64),up(64,32), nn.Conv2d(32,1,3,1,1), nn.Sigmoid())
    def forward(s,z): return s.net(z)

enc,dec,vq = Enc().to(dev), Dec().to(dev), VQ().to(dev)
ae_params = sum(p.numel() for m in (enc,dec,vq) for p in m.parameters())
print("AE+VQ params:", ae_params)

# frames for AE training
frames = X[:, :, None].reshape(S*T, 1, H, W)
nval = int(0.1*len(frames)); perm = torch.randperm(len(frames))
val_idx, tr_idx = perm[:nval], perm[nval:]
opt = torch.optim.AdamW(list(enc.parameters())+list(dec.parameters())+list(vq.parameters()), lr=2e-4)
AE_EPOCHS = int(os.environ.get("AE_EPOCHS", 80)); BS=32
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=AE_EPOCHS)   # cosine LR schedule
def batches(idx):
    for i in range(0,len(idx),BS): yield idx[i:i+BS]
t0=time.time(); ae_log=[]
for ep in range(AE_EPOCHS):
    enc.train();dec.train();vq.train()
    for b in batches(tr_idx[torch.randperm(len(tr_idx))]):
        x=frames[b].to(dev)
        if torch.rand(1).item()<0.5: x=torch.flip(x,[3])
        z=enc(x); zq,vql,_=vq(z); xr=dec(zq)
        loss=F.mse_loss(xr,x)+0.3*F.l1_loss(xr,x)+vql
        if LPIPS is not None:                       # perceptual loss on 3-ch replicated input
            loss=loss+LPIPS_W*LPIPS(xr.repeat(1,3,1,1)*2-1, x.repeat(1,3,1,1)*2-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
    if ep%10==0 or ep==AE_EPOCHS-1:
        enc.eval();dec.eval();vq.eval()
        with torch.no_grad():
            xv=frames[val_idx].to(dev); zq,_,_=vq(enc(xv)); xr=dec(zq)
            vmse=F.mse_loss(xr,xv).item()
        ae_log.append({"epoch":ep,"val_mse":vmse,"sec":round(time.time()-t0)})
        print(f"[AE] ep{ep} val_mse={vmse:.5f} ({round(time.time()-t0)}s)", flush=True)
torch.save({"enc":enc.state_dict(),"dec":dec.state_dict(),"vq":vq.state_dict()}, f"{OUT}/wm_v2_ae_gpu.pt")

# ---------- encode all sequences to latent tokens ----------
enc.eval();vq.eval()
with torch.no_grad():
    Z=[]
    for s in range(S):
        zs=enc(X[s][:,None].to(dev)); zq,_,idx=vq(zs)   # idx: (T,16,16)
        Z.append(zq.cpu())
    Z=torch.stack(Z)                                    # (S,T,D,16,16)
print("latents:", Z.shape)
Zn=(Z-Z.mean())/ (Z.std()+1e-6)
D=Z.shape[2]; hh=Z.shape[3]

# ---------- Transformer dynamics + latent action ----------
class Action(nn.Module):
    def __init__(s,D,adim=8):
        super().__init__()
        s.net=nn.Sequential(nn.Conv2d(2*D,64,3,1,1),nn.SiLU(),nn.Conv2d(64,64,3,1,1),nn.SiLU(),nn.AdaptiveAvgPool2d(1))
        s.fc=nn.Linear(64,adim)
    def forward(s,zt,zt1): return s.fc(s.net(torch.cat([zt,zt1],1)).flatten(1))
class Dyn(nn.Module):
    def __init__(s,D,hh,adim=8,dmodel=256,nlayer=4):
        super().__init__()
        s.tok=nn.Linear(D,dmodel); s.pos=nn.Parameter(torch.randn(1,hh*hh,dmodel)*0.02)
        s.afc=nn.Linear(adim,dmodel)
        s.tr=nn.TransformerEncoder(nn.TransformerEncoderLayer(dmodel,8,dmodel*2,batch_first=True,activation="gelu"),nlayer)
        s.out=nn.Linear(dmodel,D); s.D,s.hh=D,hh
    def forward(s,zt,a):                    # zt:(B,D,h,w) a:(B,adim)
        B=zt.shape[0]; x=zt.flatten(2).transpose(1,2)      # (B,h*w,D)
        h=s.tok(x)+s.pos+s.afc(a)[:,None]
        h=s.tr(h); d=s.out(h).transpose(1,2).view(B,s.D,s.hh,s.hh)
        return zt+d                          # residual
A=Action(D).to(dev); Dm=Dyn(D,hh).to(dev)
dyn_params=sum(p.numel() for m in (A,Dm) for p in m.parameters()); print("dyn+action params:",dyn_params)

# training windows: predict z_{t+1} from z_t + action(z_t,z_{t+1})
pairs=[(s,t) for s in range(S) for t in range(TRAIN_T-1)]
optd=torch.optim.AdamW(list(A.parameters())+list(Dm.parameters()),lr=3e-4)
DYN_EPOCHS=int(os.environ.get("DYN_EPOCHS",300)); dyn_log=[]; t0=time.time()
Zn=Zn.to(dev)
for ep in range(DYN_EPOCHS):
    np.random.shuffle(pairs); tot=0
    for i in range(0,len(pairs),64):
        bp=pairs[i:i+64]
        zt=torch.stack([Zn[s,t] for s,t in bp]); zt1=torch.stack([Zn[s,t+1] for s,t in bp])
        noise=torch.rand(len(bp),1,1,1,device=dev)*0.3            # diffusion-forcing
        a=A(zt,zt1); pred=Dm(zt+noise*torch.randn_like(zt),a)
        loss=F.mse_loss(pred,zt1)+1e-3*(a**2).mean()
        optd.zero_grad();loss.backward();optd.step();tot+=loss.item()
    if ep%50==0 or ep==DYN_EPOCHS-1:
        dyn_log.append({"epoch":ep,"loss":tot/(len(pairs)//64+1),"sec":round(time.time()-t0)})
        print(f"[DYN] ep{ep} loss={tot/(len(pairs)//64+1):.4f} ({round(time.time()-t0)}s)",flush=True)
torch.save({"A":A.state_dict(),"D":Dm.state_dict()}, f"{OUT}/wm_v2_dyn_gpu.pt")

# ---------- held-out evaluation: autoregressive rollout vs baselines ----------
A.eval();Dm.eval()
def decode(z): 
    with torch.no_grad(): return dec((z*Z.std().to(dev)+Z.mean().to(dev))).cpu()
mse_model=mse_persist=mse_lin=0; n=0
with torch.no_grad():
    for s in range(S):
        # mean action trajectory as expected-dynamics prior
        acts=[A(Zn[s,t:t+1],Zn[s,t+1:t+2]) for t in range(TRAIN_T-1)]
        amean=torch.stack(acts).mean(0)
        z=Zn[s,TRAIN_T-1:TRAIN_T].clone()
        for t in range(TRAIN_T,T):
            z=Dm(z,amean)
            real=X[s,t][None,None].to(dev); pred=decode(z)[:, :1].to(dev)
            persist=X[s,TRAIN_T-1][None,None].to(dev)
            lin=(2*X[s,TRAIN_T-1]-X[s,TRAIN_T-2])[None,None].clamp(0,1).to(dev)
            mse_model+=F.mse_loss(pred,real).item(); mse_persist+=F.mse_loss(persist,real).item()
            mse_lin+=F.mse_loss(lin,real).item(); n+=1
res={"ae_params":ae_params,"dyn_params":dyn_params,"device":torch.cuda.get_device_name() if dev=="cuda" else "cpu",
     "mse_model_ar":mse_model/n,"mse_persist":mse_persist/n,"mse_linear":mse_lin/n,
     "improve_vs_persist_pct":100*(mse_persist-mse_model)/mse_persist,
     "ae_log":ae_log,"dyn_log":dyn_log,"resolution":256,"tokenizer":"VQ-VAE","dynamics":"Transformer"}
json.dump(res,open(f"{OUT}/wm_v2_gpu_eval.json","w"),indent=2)
print("RESULT:",json.dumps({k:res[k] for k in ["mse_model_ar","mse_persist","improve_vs_persist_pct"]}))
print("done")
