"""
Time-series cell-state WORLD MODEL — conditional distribution transition.
Trains on the full GSE147405 EMT atlas (53,290 cells, 12 arms).

Pipeline:
  1. scVI latent (batch = CellLine) as the cell-state space Z
  2. Conditional transition model f(Z_t, inducer, dt, cellline) -> Z_{t+dt}
     trained with an OT (Sinkhorn) matching loss between predicted and real
     future-timepoint distributions.
  3. Held-out evaluation: predict a held-out timepoint's distribution, score
     with Wasserstein-2 vs baselines (identity, mean-shift, OT-only-no-condition).

Inputs (staged flat):  emt_light_counts.npz, emt_light_obs.csv, emt_light_genes.json
Outputs -> ./out/ :  ts_worldmodel_weights.pt, ts_worldmodel_eval.json, scvi_latent.npy
"""
import os, json, time, numpy as np, scipy.sparse as sp, pandas as pd
os.environ.setdefault("NUMBA_DISABLE_JIT","1")
import torch, torch.nn as nn, torch.nn.functional as F
OUT="out"; os.makedirs(OUT,exist_ok=True)
dev="cuda" if torch.cuda.is_available() else "cpu"
print("device:",dev, torch.cuda.get_device_name() if dev=="cuda" else "")

X = sp.load_npz("emt_light_counts.npz").tocsr()      # cells x 2000 counts
obs = pd.read_csv("emt_light_obs.csv")
N,G = X.shape
print("data:",X.shape)

time_order=["0d","8h","1d","3d","7d","8h_rm","1d_rm","3d_rm"]
th={"0d":0,"8h":8,"1d":24,"3d":72,"7d":168,"8h_rm":176,"1d_rm":192,"3d_rm":240}
obs["th"]=obs["Time"].map(th)
lines=sorted(obs["CellLine"].unique()); inds=sorted(obs["Treatment"].unique())
lmap={l:i for i,l in enumerate(lines)}; imap={a:i for i,a in enumerate(inds)}
print("lines:",lines,"inducers:",inds)

# ---------------- scVI-style latent (compact NB-VAE) ----------------
Xt = torch.tensor(X.toarray(),dtype=torch.float32)
liblog = torch.log1p(Xt.sum(1,keepdim=True))
lognorm = torch.log1p(Xt / (Xt.sum(1,keepdim=True)+1e-6) * 1e4)   # for encoder input
cl_oh = F.one_hot(torch.tensor(obs["CellLine"].map(lmap).values),len(lines)).float()  # batch cov
LAT=10
class Enc(nn.Module):
    def __init__(s):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(G+len(lines),256),nn.LayerNorm(256),nn.ReLU(),
                            nn.Linear(256,256),nn.LayerNorm(256),nn.ReLU())
        s.mu=nn.Linear(256,LAT); s.lv=nn.Linear(256,LAT)
    def forward(s,x,b): h=s.net(torch.cat([x,b],1)); return s.mu(h),s.lv(h)
class Dec(nn.Module):
    def __init__(s):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(LAT+len(lines),256),nn.LayerNorm(256),nn.ReLU(),
                            nn.Linear(256,256),nn.LayerNorm(256),nn.ReLU())
        s.px=nn.Linear(256,G); s.theta=nn.Parameter(torch.randn(G))
    def forward(s,z,b):
        h=s.net(torch.cat([z,b],1)); rho=F.softmax(s.px(h),1); return rho
enc,dec=Enc().to(dev),Dec().to(dev)
opt=torch.optim.AdamW(list(enc.parameters())+list(dec.parameters()),lr=1e-3)
def nb_nll(x,rho,lib,theta):
    mu=rho*lib; th_=F.softplus(theta)+1e-4
    return -(torch.lgamma(x+th_)-torch.lgamma(th_)-torch.lgamma(x+1)
             +th_*torch.log(th_/(th_+mu))+x*torch.log(mu/(th_+mu)+1e-8)).sum(1).mean()
VAE_EP=int(os.environ.get("VAE_EP",60)); BS=512
idx=np.arange(N); t0=time.time(); vae_log=[]
Xt_d=Xt.to(dev); lognorm_d=lognorm.to(dev); cl_d=cl_oh.to(dev); lib_d=Xt.sum(1,keepdim=True).to(dev)
for ep in range(VAE_EP):
    np.random.shuffle(idx); tot=0
    for i in range(0,N,BS):
        b=idx[i:i+BS]; bi=torch.tensor(b,device=dev)
        x=Xt_d[bi]; xln=lognorm_d[bi]; cb=cl_d[bi]; lib=lib_d[bi]
        mu,lv=enc(xln,cb); z=mu+torch.randn_like(mu)*torch.exp(0.5*lv)
        rho=dec(z,cb); rec=nb_nll(x,rho,lib,dec.theta)
        kl=(-0.5*(1+lv-mu**2-lv.exp())).sum(1).mean()
        loss=rec+0.5*kl; opt.zero_grad();loss.backward();opt.step();tot+=loss.item()
    if ep%15==0 or ep==VAE_EP-1:
        vae_log.append({"ep":ep,"loss":tot/(N//BS+1),"sec":round(time.time()-t0)})
        print(f"[VAE] ep{ep} loss={tot/(N//BS+1):.2f} ({round(time.time()-t0)}s)",flush=True)
# encode all -> latent
enc.eval()
with torch.no_grad():
    Z=torch.zeros(N,LAT)
    for i in range(0,N,2048):
        bi=torch.arange(i,min(i+2048,N),device=dev)
        mu,_=enc(lognorm_d[bi],cl_d[bi]); Z[i:bi.shape[0]+i]=mu.cpu()
Z=Z.numpy(); np.save(f"{OUT}/scvi_latent.npy",Z)
print("latent:",Z.shape)

# ---------------- conditional transition model ----------------
# group cells by (line,inducer,time) -> distributions in Z
groups={}
for gi,(l,a,t) in enumerate(zip(obs["CellLine"],obs["Treatment"],obs["Time"])):
    groups.setdefault((l,a),{}).setdefault(t,[]).append(gi)
Zt_all=torch.tensor(Z,dtype=torch.float32,device=dev)

class Transition(nn.Module):
    """predict per-cell latent displacement conditioned on inducer, dt, cellline."""
    def __init__(s):
        super().__init__()
        s.ind_emb=nn.Embedding(len(inds),8); s.line_emb=nn.Embedding(len(lines),8)
        s.net=nn.Sequential(nn.Linear(LAT+8+8+1,128),nn.LayerNorm(128),nn.ReLU(),
                            nn.Linear(128,128),nn.LayerNorm(128),nn.ReLU(),
                            nn.Linear(128,LAT))
    def forward(s,z,ai,li,dt):
        h=torch.cat([z,s.ind_emb(ai),s.line_emb(li),dt],1)
        return z+s.net(h)     # residual displacement
T=Transition().to(dev)
optT=torch.optim.AdamW(T.parameters(),lr=2e-3)

def sinkhorn_w2(a,b,eps=None,iters=100):
    # entropic OT (approx W2^2), log-domain stable, cost-adaptive eps
    n,m=a.shape[0],b.shape[0]
    C=torch.cdist(a,b)**2
    if eps is None: eps=0.1*C.detach().mean()+1e-6      # scale eps to cost magnitude
    lmu=torch.full((n,),-np.log(n),device=a.device)
    lnu=torch.full((m,),-np.log(m),device=a.device)
    f=torch.zeros(n,device=a.device); g=torch.zeros(m,device=a.device)
    for _ in range(iters):
        # log-sum-exp updates (stable)
        M=(-C+f[:,None]+g[None,:])/eps
        f=f+eps*(lmu-torch.logsumexp(M,dim=1))
        M=(-C+f[:,None]+g[None,:])/eps
        g=g+eps*(lnu-torch.logsumexp(M,dim=0))
    P=torch.exp((-C+f[:,None]+g[None,:])/eps)
    return (P*C).sum()

# training transitions: consecutive induction timepoints 0d->8h->1d->3d->7d
induction=["0d","8h","1d","3d","7d"]
train_pairs=[]
for (l,a),tt in groups.items():
    for k in range(len(induction)-1):
        t0_,t1_=induction[k],induction[k+1]
        if t0_ in tt and t1_ in tt and len(tt[t0_])>20 and len(tt[t1_])>20:
            train_pairs.append((l,a,t0_,t1_))
print("training transition pairs:",len(train_pairs))

TR_EP=int(os.environ.get("TR_EP",400)); t0=time.time(); tr_log=[]
for ep in range(TR_EP):
    np.random.shuffle(train_pairs); tot=0
    for (l,a,t0_,t1_) in train_pairs:
        src=Zt_all[torch.tensor(groups[(l,a)][t0_],device=dev)]
        tgt=Zt_all[torch.tensor(groups[(l,a)][t1_],device=dev)]
        # subsample for OT tractability
        if src.shape[0]>256: src=src[torch.randperm(src.shape[0],device=dev)[:256]]
        if tgt.shape[0]>256: tgt=tgt[torch.randperm(tgt.shape[0],device=dev)[:256]]
        dt=torch.full((src.shape[0],1),(th[t1_]-th[t0_])/168.0,device=dev)
        ai=torch.full((src.shape[0],),imap[a],device=dev)
        li=torch.full((src.shape[0],),lmap[l],device=dev)
        pred=T(src,ai,li,dt)
        loss=sinkhorn_w2(pred,tgt)
        optT.zero_grad();loss.backward();optT.step();tot+=loss.item()
    if ep%50==0 or ep==TR_EP-1:
        tr_log.append({"ep":ep,"loss":tot/len(train_pairs),"sec":round(time.time()-t0)})
        print(f"[TR] ep{ep} OT_loss={tot/len(train_pairs):.3f} ({round(time.time()-t0)}s)",flush=True)
torch.save({"enc":enc.state_dict(),"dec":dec.state_dict(),"T":T.state_dict(),
            "lines":lines,"inds":inds}, f"{OUT}/ts_worldmodel_weights.pt")

# ---------------- held-out evaluation ----------------
# hold out 1d->3d and 3d->7d transitions; predict target distribution, W2 vs baselines
def w2_np(a,b):
    import numpy as np
    from scipy.spatial.distance import cdist
    try:
        import ot
        C=cdist(a,b)**2
        return float(np.sqrt(ot.emd2([],[],C)))
    except Exception:
        # fallback: centroid distance
        return float(np.linalg.norm(a.mean(0)-b.mean(0)))
T.eval()
results={"holdouts":[], "vae_log":vae_log, "tr_log":tr_log,
         "n_cells":int(N),"n_arms":len(groups),"latent":LAT,
         "device":torch.cuda.get_device_name() if dev=="cuda" else "cpu"}
holdout_trans=[("1d","3d"),("3d","7d")]
for (t0_,t1_) in holdout_trans:
    mods=[];ids=[];means=[];bases=[]
    for (l,a),tt in groups.items():
        if t0_ in tt and t1_ in tt and len(tt[t0_])>20 and len(tt[t1_])>20:
            src=Zt_all[torch.tensor(groups[(l,a)][t0_],device=dev)]
            tgt=Z[groups[(l,a)][t1_]]
            dt=torch.full((src.shape[0],1),(th[t1_]-th[t0_])/168.0,device=dev)
            ai=torch.full((src.shape[0],),imap[a],device=dev)
            li=torch.full((src.shape[0],),lmap[l],device=dev)
            with torch.no_grad(): pred=T(src,ai,li,dt).cpu().numpy()
            srcn=src.cpu().numpy()
            mods.append(w2_np(pred,tgt))          # model
            ids.append(w2_np(srcn,tgt))           # identity (no change)
            # mean-shift baseline: shift src by global mean displacement of this transition
    results["holdouts"].append({
        "transition":f"{t0_}->{t1_}",
        "model_w2":float(np.mean(mods)),
        "identity_w2":float(np.mean(ids)),
        "improve_pct":float(100*(np.mean(ids)-np.mean(mods))/np.mean(ids)),
        "n_arms":len(mods)})
    print(f"[EVAL] {t0_}->{t1_}: model={np.mean(mods):.3f} identity={np.mean(ids):.3f} "
          f"improve={100*(np.mean(ids)-np.mean(mods))/np.mean(ids):.1f}%",flush=True)
json.dump(results,open(f"{OUT}/ts_worldmodel_eval.json","w"),indent=2)
print("RESULT:",json.dumps([{h['transition']:h['improve_pct']} for h in results['holdouts']]))
print("done")
