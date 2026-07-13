"""Load the intermediate EMA checkpoint from Volume, generate S-conditioned samples +
guidance sweep on val cells. Read-only on the training job (separate container).
Outputs ./out/: diffS_test_samples.npz, diffS_test_sweep.json"""
import os, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
dev="cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("./out",exist_ok=True)
T=400; SDIM=128

d=np.load("/data/xenium_unified.npz",allow_pickle=True)
expr=torch.tensor(d["expr"].astype(np.float32))
patches=torch.tensor(d["patches_u8"].astype(np.float32)/255.0).unsqueeze(1)*2-1
va=d["val_idx"]; E_DIM=expr.shape[1]

class Encoder(nn.Module):
    def __init__(self,e,s):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(e,512),nn.GELU(),nn.LayerNorm(512),
            nn.Linear(512,256),nn.GELU(),nn.LayerNorm(256),nn.Linear(256,s))
    def forward(self,x): return self.net(x)
ck=torch.load("unified_state_model.pt",map_location=dev)
enc=Encoder(E_DIM,SDIM).to(dev); enc.load_state_dict(ck["enc"]); enc.eval()
for p in enc.parameters(): p.requires_grad=False

def cosine_betas(T,s=0.008):
    x=torch.linspace(0,T,T+1); ac=torch.cos(((x/T)+s)/(1+s)*math.pi/2)**2; ac=ac/ac[0]
    return (1-(ac[1:]/ac[:-1])).clamp(1e-4,0.999)
betas=cosine_betas(T); acp=torch.cumprod(1-betas,0).to(dev)
def temb(t,dim=256):
    half=dim//2; f=torch.exp(-math.log(10000)*torch.arange(half,device=t.device)/half)
    a=t[:,None].float()*f[None]; return torch.cat([a.sin(),a.cos()],-1)
class FiLM(nn.Module):
    def __init__(self,cdim,ch): super().__init__(); self.f=nn.Linear(cdim,ch*2)
    def forward(self,h,c): g,b=self.f(c)[:,:,None,None].chunk(2,1); return h*(1+g)+b
class ResBlock(nn.Module):
    def __init__(self,i,o,cdim):
        super().__init__()
        self.c1=nn.Conv2d(i,o,3,1,1); self.n1=nn.GroupNorm(8,o)
        self.c2=nn.Conv2d(o,o,3,1,1); self.n2=nn.GroupNorm(8,o)
        self.film=FiLM(cdim,o); self.skip=nn.Conv2d(i,o,1) if i!=o else nn.Identity()
    def forward(self,x,c):
        h=F.silu(self.n1(self.c1(x))); h=self.film(h,c); h=F.silu(self.n2(self.c2(h)))
        return h+self.skip(x)
class SUNet(nn.Module):
    def __init__(self,sdim,base=96,cdim=256):
        super().__init__()
        self.temb=nn.Sequential(nn.Linear(256,cdim),nn.SiLU(),nn.Linear(cdim,cdim))
        self.semb=nn.Sequential(nn.Linear(sdim,cdim),nn.SiLU(),nn.Linear(cdim,cdim))
        self.null=nn.Parameter(torch.zeros(cdim))
        self.inp=nn.Conv2d(1,base,3,1,1)
        self.d1=ResBlock(base,base,cdim); self.d2=ResBlock(base,base*2,cdim); self.d3=ResBlock(base*2,base*4,cdim)
        self.mid=ResBlock(base*4,base*4,cdim)
        self.u3=ResBlock(base*4+base*4,base*2,cdim); self.u2=ResBlock(base*2+base*2,base,cdim); self.u1=ResBlock(base+base,base,cdim)
        self.out=nn.Sequential(nn.GroupNorm(8,base),nn.SiLU(),nn.Conv2d(base,1,3,1,1))
        self.pool=nn.AvgPool2d(2)
    def forward(self,x,t,s,drop=None):
        c=self.temb(temb(t)); sc=self.semb(s)
        if drop is not None: sc=torch.where(drop[:,None],self.null[None].expand_as(sc),sc)
        c=c+sc
        h0=self.inp(x); h1=self.d1(h0,c); h2=self.d2(self.pool(h1),c); h3=self.d3(self.pool(h2),c)
        m=self.mid(self.pool(h3),c)
        u=self.u3(torch.cat([F.interpolate(m,scale_factor=2),h3],1),c)
        u=self.u2(torch.cat([F.interpolate(u,scale_factor=2),h2],1),c)
        u=self.u1(torch.cat([F.interpolate(u,scale_factor=2),h1],1),c)
        return self.out(u)

model=SUNet(SDIM).to(dev)
ema=torch.load("/data/diffusion_S_ckpt.pt",map_location=dev)
model.load_state_dict(ema); model.eval()
print("loaded intermediate ckpt",flush=True)

with torch.no_grad():
    S=torch.cat([enc(expr[i:i+8192].to(dev)).cpu() for i in range(0,len(expr),8192)]).to(dev)
STEPS=50; ts=torch.linspace(T-1,0,STEPS).long().to(dev)
@torch.no_grad()
def ddim(s,w):
    B=len(s); x=torch.randn(B,1,64,64,device=dev)
    for k in range(STEPS):
        t=ts[k].expand(B)
        ec=model(x,t,s,drop=torch.zeros(B,dtype=torch.bool,device=dev))
        un=model(x,t,s,drop=torch.ones(B,dtype=torch.bool,device=dev))
        eps=un+w*(ec-un); ac=acp[ts[k]]; x0=((x-(1-ac).sqrt()*eps)/ac.sqrt()).clamp(-1,1)
        x=(acp[ts[k+1]].sqrt()*x0+(1-acp[ts[k+1]]).sqrt()*eps) if k<STEPS-1 else x0
    return x
vsel=va[:128]; s=S[torch.tensor(vsel,device=dev)]; real=patches[torch.tensor(vsel)].to(dev)
perm=np.random.default_rng(0).permutation(len(vsel))
sweep={}
for w in [0.0,1.0,2.0,3.0,5.0]:
    smp=ddim(s,w)
    mt=F.mse_loss(smp,real).item(); ms=F.mse_loss(smp,real[perm]).item()
    lap=(smp[:,:,1:,:]-smp[:,:,:-1,:]).abs().mean().item()+(smp[:,:,:,1:]-smp[:,:,:,:-1]).abs().mean().item()
    sweep[f"w{w}"]=dict(mse_true=mt,mse_shuffle=ms,sharp=lap,cell_specific=mt<ms)
    print(f"w={w}: true={mt:.4f} shuf={ms:.4f} sharp={lap:.4f} cell_specific={mt<ms}",flush=True)
# generate at w=2 and w=3 for viewing
best3=ddim(s,3.0).cpu().numpy(); best2=ddim(s,2.0).cpu().numpy()
np.savez("./out/diffS_test_samples.npz",samples=((best3+1)/2),samples_w2=((best2+1)/2),
    real=((real.cpu().numpy()+1)/2))
json.dump(sweep,open("./out/diffS_test_sweep.json","w"),indent=2)
print("DONE test",flush=True)
