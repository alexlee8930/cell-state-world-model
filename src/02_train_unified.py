"""Unified cell-state world model: ONE shared encoder E (expr->S 128d),
three task heads sharing S: (1) expr reconstruction, (2) morphology (DINOv2-384) regression,
(3) spatial GNN (neighbor S -> center S). Joint training on 200k cells from Volume.
Honest validation: each task vs shuffle control. Saves encoder+heads + latent S for all cells.
Outputs ./out/: unified_state_model.pt, unified_train_log.json, unified_eval.json, latent_S.npz(small: val only)
"""
import os, json, time, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
dev="cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("./out",exist_ok=True)
EPOCHS=int(os.environ.get("EPOCHS","100")); BS=int(os.environ.get("BS","1024"))
SDIM=128; KNN=12

d=np.load("/data/xenium_unified.npz",allow_pickle=True)
expr=torch.tensor(d["expr"].astype(np.float32))           # (N,422)
dino=torch.tensor(d["dino"].astype(np.float32))            # (N,384)
xy=d["xy"].astype(np.float32)                              # (N,2)
tr=d["tr_idx"]; va=d["val_idx"]; N,E_DIM=expr.shape
# standardize dino targets
dmu=dino[torch.tensor(tr)].mean(0); dsd=dino[torch.tensor(tr)].std(0)+1e-6
dino=(dino-dmu)/dsd
print(f"N={N} E_DIM={E_DIM} dino={dino.shape[1]} train={len(tr)} dev={dev}",flush=True)

# spatial kNN graph (on all cells, via sklearn on CPU once)
from sklearn.neighbors import NearestNeighbors
t0=time.time()
nn_=NearestNeighbors(n_neighbors=KNN+1).fit(xy)
_,knn=nn_.kneighbors(xy); knn=knn[:,1:]  # (N,KNN) exclude self
knn=torch.tensor(knn,dtype=torch.long)
print("kNN graph built",round(time.time()-t0),"s",flush=True)

class Encoder(nn.Module):
    def __init__(self,e,s):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(e,512),nn.GELU(),nn.LayerNorm(512),
            nn.Linear(512,256),nn.GELU(),nn.LayerNorm(256),nn.Linear(256,s))
    def forward(self,x): return self.net(x)
class Head(nn.Module):
    def __init__(self,s,o,h=256):
        super().__init__(); self.net=nn.Sequential(nn.Linear(s,h),nn.GELU(),nn.Linear(h,o))
    def forward(self,x): return self.net(x)

enc=Encoder(E_DIM,SDIM).to(dev)
head_expr=Head(SDIM,E_DIM).to(dev)     # reconstruction
head_morph=Head(SDIM,dino.shape[1]).to(dev)  # morphology
head_spatial=Head(SDIM,SDIM).to(dev)   # neighbor-agg S -> center S
params=list(enc.parameters())+list(head_expr.parameters())+list(head_morph.parameters())+list(head_spatial.parameters())
nparam=sum(p.numel() for p in params); print("params",nparam,flush=True)
opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=1e-5)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
scaler=torch.cuda.amp.GradScaler(enabled=(dev=="cuda"))

expr_d=expr.to(dev); dino_d=dino.to(dev); knn_d=knn.to(dev)
LR,LM,LS=1.0,1.0,0.5
log=[]; t0=time.time()
for ep in range(EPOCHS):
    enc.train(); idx=np.random.permutation(tr); tl=dict(r=0,m=0,s=0); nb=0
    for i in range(0,len(idx),BS):
        bi=torch.tensor(idx[i:i+BS],device=dev)
        e=expr_d[bi]
        with torch.cuda.amp.autocast(enabled=(dev=="cuda")):
            S=enc(e)
            # 1. expr recon
            lr_=F.mse_loss(head_expr(S),e)
            # 2. morph
            lm_=F.mse_loss(head_morph(S),dino_d[bi])
            # 3. spatial: neighbor expr -> S, aggregate, predict center S
            nb_idx=knn_d[bi]                       # (B,KNN)
            with torch.no_grad(): pass
            Sneigh=enc(expr_d[nb_idx.reshape(-1)]).reshape(len(bi),KNN,SDIM).mean(1)
            ls_=F.mse_loss(head_spatial(Sneigh),S.detach())
            loss=LR*lr_+LM*lm_+LS*ls_
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tl["r"]+=lr_.item(); tl["m"]+=lm_.item(); tl["s"]+=ls_.item(); nb+=1
    sched.step()
    if ep%5==0 or ep==EPOCHS-1:
        log.append(dict(epoch=ep,recon=tl["r"]/nb,morph=tl["m"]/nb,spatial=tl["s"]/nb,t=round(time.time()-t0)))
        print(f"ep{ep} recon={tl['r']/nb:.4f} morph={tl['m']/nb:.4f} spatial={tl['s']/nb:.4f} {round(time.time()-t0)}s",flush=True)

# ---- eval on val (+ shuffle controls) ----
enc.eval()
@torch.no_grad()
def encode(ix):
    out=[]
    for i in range(0,len(ix),4096):
        out.append(enc(expr_d[torch.tensor(ix[i:i+4096],device=dev)]).cpu())
    return torch.cat(out)
Sval=encode(va)
Sva_d=Sval.to(dev)
va_t=torch.tensor(va,device=dev)
with torch.no_grad():
    # morph R2 + shuffle
    pm=head_morph(Sva_d).cpu().numpy(); tm=dino[torch.tensor(va)].numpy()
    def r2(p,t): 
        ss_res=((p-t)**2).sum(); ss_tot=((t-t.mean(0))**2).sum(); return 1-ss_res/ss_tot
    morph_r2=float(r2(pm,tm))
    perm=np.random.permutation(len(va)); morph_r2_shuf=float(r2(pm,tm[perm]))
    # expr recon R2
    pe=head_expr(Sva_d).cpu().numpy(); te=expr[torch.tensor(va)].numpy()
    recon_r2=float(r2(pe,te)); recon_r2_shuf=float(r2(pe,te[perm]))
    # spatial: predict center S from neighbors, R2 vs shuffle
    nb_idx=knn_d[va_t]
    Sn=enc(expr_d[nb_idx.reshape(-1)]).reshape(len(va),KNN,SDIM).mean(1)
    ps=head_spatial(Sn).cpu().numpy(); ts=Sval.numpy()
    spatial_r2=float(r2(ps,ts)); spatial_r2_shuf=float(r2(ps,ts[perm]))
eval_=dict(morph_r2=morph_r2,morph_r2_shuffle=morph_r2_shuf,
    recon_r2=recon_r2,recon_r2_shuffle=recon_r2_shuf,
    spatial_r2=spatial_r2,spatial_r2_shuffle=spatial_r2_shuf,nparam=int(nparam))
print("EVAL",json.dumps(eval_,indent=2),flush=True)

torch.save(dict(enc=enc.state_dict(),head_expr=head_expr.state_dict(),
    head_morph=head_morph.state_dict(),head_spatial=head_spatial.state_dict(),
    dmu=dmu,dsd=dsd,SDIM=SDIM),"./out/unified_state_model.pt")
json.dump(log,open("./out/unified_train_log.json","w"),indent=2)
json.dump(eval_,open("./out/unified_eval.json","w"),indent=2)
# save val latent S (small) for downstream figures
np.savez("./out/latent_S.npz",S_val=Sval.numpy(),val_idx=va,xy_val=xy[va])
print("DONE",round(time.time()-t0),"s",flush=True)
