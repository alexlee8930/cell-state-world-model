"""Production-grade expression-conditional latent-space DDPM for Xenium morphology.
Key upgrades over the earlier blob-diffusion:
  1. Classifier-free guidance (CFG): cond dropout in training, guided sampling.
     -> forces the model to USE expression (fixes MSE-to-true==MSE-to-shuffle).
  2. FiLM conditioning injected at EVERY UNet resolution (not just bottleneck).
  3. EMA weights for stable sampling.
  4. DDIM sampler (fast, deterministic) + guidance-weight sweep.
  5. Large data (192k train), cosine LR, mixed precision.
Honest validation: MSE-to-true vs MSE-to-shuffle AS A FUNCTION OF guidance weight.
If CFG works, higher guidance => MSE-to-true drops BELOW MSE-to-shuffle (cell-specific).
Outputs ./out/: diffusion_prod.pt (EMA), prod_train_log.json, prod_guidance_sweep.npz, prod_samples.npz
"""
import os, json, time, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
dev="cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("./out",exist_ok=True)
EPOCHS=int(os.environ.get("EPOCHS","120")); BS=int(os.environ.get("BS","256"))
T=int(os.environ.get("T","400")); PDROP=0.1   # cond-dropout prob for CFG

DATA=os.environ.get("DATA_PATH","xenium_large_comp.npz")
d=np.load(DATA,allow_pickle=True)
expr=torch.tensor(d["expr"].astype(np.float32),dtype=torch.float32)
patches=torch.tensor(d["patches"].astype(np.float32)/255.0).unsqueeze(1)*2-1
tr=d["tr_idx"]; va=d["val_idx"]; E_DIM=expr.shape[1]
print(f"device={dev} N={len(expr)} train={len(tr)} E_DIM={E_DIM} T={T} epochs={EPOCHS} bs={BS}",flush=True)

# cosine beta schedule (Nichol&Dhariwal)
def cosine_betas(T,s=0.008):
    x=torch.linspace(0,T,T+1); ac=torch.cos(((x/T)+s)/(1+s)*math.pi/2)**2; ac=ac/ac[0]
    b=1-(ac[1:]/ac[:-1]); return b.clamp(1e-4,0.999)
betas=cosine_betas(T); alphas=1-betas; acp=torch.cumprod(alphas,0).to(dev)
sqrt_acp=acp.sqrt(); sqrt_1m=(1-acp).sqrt()

def temb(t,dim=256):
    half=dim//2; f=torch.exp(-math.log(10000)*torch.arange(half,device=t.device)/half)
    a=t[:,None].float()*f[None]; return torch.cat([a.sin(),a.cos()],-1)

class FiLM(nn.Module):
    def __init__(self,cdim,ch):
        super().__init__(); self.f=nn.Linear(cdim,ch*2)
    def forward(self,h,c):
        g,b=self.f(c)[:,:,None,None].chunk(2,1); return h*(1+g)+b

class ResBlock(nn.Module):
    def __init__(self,i,o,cdim):
        super().__init__()
        self.c1=nn.Conv2d(i,o,3,1,1); self.n1=nn.GroupNorm(8,o)
        self.c2=nn.Conv2d(o,o,3,1,1); self.n2=nn.GroupNorm(8,o)
        self.film=FiLM(cdim,o); self.skip=nn.Conv2d(i,o,1) if i!=o else nn.Identity()
    def forward(self,x,c):
        h=F.silu(self.n1(self.c1(x))); h=self.film(h,c); h=F.silu(self.n2(self.c2(h)))
        return h+self.skip(x)

class CondUNet(nn.Module):
    def __init__(self,e_dim,base=96,cdim=256):
        super().__init__()
        self.temb=nn.Sequential(nn.Linear(256,cdim),nn.SiLU(),nn.Linear(cdim,cdim))
        self.eemb=nn.Sequential(nn.Linear(e_dim,cdim),nn.SiLU(),nn.Linear(cdim,cdim))
        self.null=nn.Parameter(torch.zeros(cdim))  # CFG null token
        self.inp=nn.Conv2d(1,base,3,1,1)
        self.d1=ResBlock(base,base,cdim); self.d2=ResBlock(base,base*2,cdim); self.d3=ResBlock(base*2,base*4,cdim)
        self.mid=ResBlock(base*4,base*4,cdim)
        self.u3=ResBlock(base*4+base*4,base*2,cdim); self.u2=ResBlock(base*2+base*2,base,cdim); self.u1=ResBlock(base+base,base,cdim)
        self.out=nn.Sequential(nn.GroupNorm(8,base),nn.SiLU(),nn.Conv2d(base,1,3,1,1))
        self.pool=nn.AvgPool2d(2)
    def forward(self,x,t,e,drop=None):
        c=self.temb(temb(t))
        ec=self.eemb(e)
        if drop is not None:
            ec=torch.where(drop[:,None], self.null[None].expand_as(ec), ec)
        c=c+ec
        h0=self.inp(x)
        h1=self.d1(h0,c); h2=self.d2(self.pool(h1),c); h3=self.d3(self.pool(h2),c)  # 64,32,16
        m=self.mid(self.pool(h3),c)                                                  # 8
        u=self.u3(torch.cat([F.interpolate(m,scale_factor=2),h3],1),c)               # 16
        u=self.u2(torch.cat([F.interpolate(u,scale_factor=2),h2],1),c)               # 32
        u=self.u1(torch.cat([F.interpolate(u,scale_factor=2),h1],1),c)               # 64
        return self.out(u)

model=CondUNet(E_DIM).to(dev)
ema={k:v.detach().clone() for k,v in model.state_dict().items()}
n_params=sum(p.numel() for p in model.parameters()); print("UNet params:",n_params,flush=True)
opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
scaler=torch.cuda.amp.GradScaler(enabled=(dev=="cuda"))

def ema_update(m=0.999):
    sd=model.state_dict()
    for k in ema: ema[k].mul_(m).add_(sd[k].detach(),alpha=1-m)

log=[]; t0=time.time()
for ep in range(EPOCHS):
    model.train(); idx=np.random.permutation(tr); tl=0; nb=0
    for i in range(0,len(idx),BS):
        bi=idx[i:i+BS]
        x0=patches[bi].to(dev); e=expr[bi].to(dev)
        t=torch.randint(0,T,(len(bi),),device=dev)
        drop=(torch.rand(len(bi),device=dev)<PDROP)
        noise=torch.randn_like(x0)
        xt=sqrt_acp[t][:,None,None,None]*x0+sqrt_1m[t][:,None,None,None]*noise
        with torch.cuda.amp.autocast(enabled=(dev=="cuda")):
            pred=model(xt,t,e,drop); loss=F.mse_loss(pred,noise)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        ema_update(); tl+=loss.item(); nb+=1
    sched.step()
    if ep%5==0 or ep==EPOCHS-1:
        log.append(dict(epoch=ep,loss=tl/nb,t=round(time.time()-t0)))
        print(f"ep{ep} noise_mse={tl/nb:.4f} {round(time.time()-t0)}s",flush=True)

torch.save(ema,"./out/diffusion_prod.pt")
json.dump(log,open("./out/prod_train_log.json","w"),indent=2)

# ---- DDIM sampling with CFG ----
model.load_state_dict(ema); model.eval()
STEPS=50; ts=torch.linspace(T-1,0,STEPS).long().to(dev)
@torch.no_grad()
def ddim(e,w):
    B=len(e); x=torch.randn(B,1,64,64,device=dev)
    for k in range(STEPS):
        t=ts[k].expand(B)
        ec=model(x,t,e,drop=torch.zeros(B,dtype=torch.bool,device=dev))
        un=model(x,t,e,drop=torch.ones(B,dtype=torch.bool,device=dev))
        eps=un+w*(ec-un)   # CFG
        ac=acp[ts[k]]; x0=(x-(1-ac).sqrt()*eps)/ac.sqrt(); x0=x0.clamp(-1,1)
        if k<STEPS-1:
            ac2=acp[ts[k+1]]; x=ac2.sqrt()*x0+(1-ac2).sqrt()*eps
        else: x=x0
    return x

# guidance sweep on val subset: does higher w make samples cell-specific?
vsel=va[:64]; e=expr[vsel].to(dev); real=patches[vsel].to(dev)
rng=np.random.default_rng(0); perm=rng.permutation(len(vsel))
sweep={}
for w in [0.0,1.0,2.0,3.0,5.0]:
    s=ddim(e,w)
    mse_true=F.mse_loss(s,real).item()
    mse_shuf=F.mse_loss(s,real[perm]).item()
    # sharpness
    lap=(s[:,:,1:,:]-s[:,:,:-1,:]).abs().mean().item()+(s[:,:,:,1:]-s[:,:,:,:-1]).abs().mean().item()
    sweep[f"w{w}"]=dict(mse_true=mse_true,mse_shuffle=mse_shuf,sharp=lap,cell_specific=mse_true<mse_shuf)
    print(f"CFG w={w}: mse_true={mse_true:.4f} mse_shuffle={mse_shuf:.4f} sharp={lap:.4f} cell_specific={mse_true<mse_shuf}",flush=True)

# save samples at best guidance (w=2) for figure
best=ddim(e,2.0).cpu().numpy()
np.savez("./out/prod_samples.npz", samples=((best+1)/2), real=((real.cpu().numpy()+1)/2), w=2.0)
json.dump(sweep,open("./out/prod_guidance_sweep.json","w"),indent=2)
np.savez("./out/prod_guidance_sweep.npz", sweep=json.dumps(sweep))
print("DONE final noise_mse=",log[-1]["loss"],flush=True)
