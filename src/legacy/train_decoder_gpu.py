"""Expression -> morphology conditional decoder. Trains on Xenium CRC pairs.
Loss = MSE + 0.3*L1 + LPIPS_W*LPIPS(vgg). Held-out eval vs baselines in-script.
Outputs to ./out/: xenium_decoder.pt, decoder_train_log.json, decoder_eval.json
"""
import os, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
OUT="./out"; os.makedirs(OUT, exist_ok=True)
DATA=os.environ.get("DATA_PATH","xenium_pairs.npz")
EPOCHS=int(os.environ.get("EPOCHS","120"))
LPIPS_W=float(os.environ.get("LPIPS_W","0.5"))
BS=256

d=np.load(DATA, allow_pickle=True)
expr=torch.tensor(d["expr"],dtype=torch.float32)          # (N,256)
if "patches_u8" in d.files:
    patches=torch.tensor(d["patches_u8"].astype(np.float32)/255.0).unsqueeze(1)  # (N,1,64,64)
else:
    patches=torch.tensor(d["patches"],dtype=torch.float32).unsqueeze(1)
tr=d["tr_idx"]; va=d["val_idx"]
print(f"device={dev} N={len(expr)} expr_dim={expr.shape[1]} train={len(tr)} val={len(va)} epochs={EPOCHS}", flush=True)

E_DIM=expr.shape[1]
class Decoder(nn.Module):
    def __init__(self, e_dim, ch=256):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(e_dim,768),nn.SiLU(),
                               nn.Linear(768,768),nn.SiLU(),
                               nn.Linear(768,ch*4*4))
        self.ch=ch
        def up(i,o): return nn.Sequential(nn.ConvTranspose2d(i,o,4,2,1),nn.GroupNorm(8,o),nn.SiLU(),
                                          nn.Conv2d(o,o,3,1,1),nn.GroupNorm(8,o),nn.SiLU())
        self.dec=nn.Sequential(up(ch,192),up(192,128),up(128,96),up(96,48))
        self.head=nn.Sequential(nn.Conv2d(48,1,3,1,1),nn.Sigmoid())
    def forward(self,e):
        z=self.enc(e).view(-1,self.ch,4,4)
        return self.head(self.dec(z))

model=Decoder(E_DIM).to(dev)
n_params=sum(p.numel() for p in model.parameters())
print("decoder params:", n_params, flush=True)

# LPIPS optional
lpips_fn=None
if LPIPS_W>0:
    try:
        import lpips
        lpips_fn=lpips.LPIPS(net="vgg").to(dev)
        for p in lpips_fn.parameters(): p.requires_grad=False
        print("LPIPS(vgg) enabled", flush=True)
    except Exception as ex:
        print("LPIPS unavailable, fallback to pixel only:", ex, flush=True); LPIPS_W=0.0

opt=torch.optim.AdamW(model.parameters(),lr=2e-4)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)

def batches(idx,bs,shuffle=True):
    idx=np.array(idx)
    if shuffle: np.random.shuffle(idx)
    for i in range(0,len(idx),bs): yield idx[i:i+bs]

def lpips_loss(xh,x):
    # replicate grayscale to 3ch, scale [0,1]->[-1,1]
    a=xh.repeat(1,3,1,1)*2-1; b=x.repeat(1,3,1,1)*2-1
    return lpips_fn(a,b).mean()

import copy
def val_mse_now():
    model.eval(); vm=0; vn=0
    with torch.no_grad():
        for bi in batches(va,BS,shuffle=False):
            e=expr[bi].to(dev); x=patches[bi].to(dev)
            vm+=F.mse_loss(model(e),x).item(); vn+=1
    return vm/vn

log=[]; t0=time.time(); best_val=1e9; best_state=None; patience=0; PAT=8
for ep in range(EPOCHS):
    model.train(); tl=0; nb=0
    for bi in batches(tr,BS):
        e=expr[bi].to(dev); x=patches[bi].to(dev)
        xh=model(e)
        loss=F.mse_loss(xh,x)+0.3*F.l1_loss(xh,x)
        if LPIPS_W>0: loss=loss+LPIPS_W*lpips_loss(xh,x)
        opt.zero_grad(); loss.backward(); opt.step()
        tl+=loss.item(); nb+=1
    sched.step()
    vm=val_mse_now()  # early-stop on pixel val MSE every epoch
    if vm<best_val-1e-5:
        best_val=vm; best_state=copy.deepcopy(model.state_dict()); patience=0
    else:
        patience+=1
    if ep%5==0 or ep==EPOCHS-1:
        log.append(dict(epoch=ep,train_loss=tl/nb,val_mse=vm,t=round(time.time()-t0)))
        print(f"ep{ep} train={tl/nb:.4f} val_mse={vm:.5f} best={best_val:.5f} pat={patience} {round(time.time()-t0)}s", flush=True)
    if patience>=PAT:
        print(f"early stop at ep{ep}, best_val={best_val:.5f}", flush=True); break

# restore best
if best_state is not None: model.load_state_dict(best_state)
print(f"restored best val_mse={best_val:.5f}", flush=True)
torch.save(model.state_dict(), f"{OUT}/xenium_decoder.pt")
json.dump(log, open(f"{OUT}/decoder_train_log.json","w"), indent=2)

# ---- held-out eval vs baselines ----
model.eval()
with torch.no_grad():
    Xv=patches[va].to(dev); Ev=expr[va].to(dev)
    pred=torch.cat([model(Ev[i:i+BS]) for i in range(0,len(Ev),BS)],0)
    mean_img=patches[tr].mean(0,keepdim=True).to(dev).expand_as(Xv)  # baseline a
    rng=np.random.default_rng(0)
    rand_idx=rng.choice(va,len(va),replace=False)
    rand_img=patches[rand_idx].to(dev)                                # baseline b
    shuf=rng.permutation(len(Ev))
    pred_shuf=torch.cat([model(Ev[shuf][i:i+BS]) for i in range(0,len(Ev),BS)],0)  # baseline c

    def mse(a,b): return F.mse_loss(a,b).item()
    def ssim(a,b):
        # simple global SSIM per image, averaged
        a=a.squeeze(1).cpu().numpy(); b=b.squeeze(1).cpu().numpy()
        out=[]
        for i in range(len(a)):
            x,y=a[i],b[i]; mx,my=x.mean(),y.mean(); vx,vy=x.var(),y.var(); cov=((x-mx)*(y-my)).mean()
            c1,c2=0.01**2,0.03**2
            out.append(((2*mx*my+c1)*(2*cov+c2))/((mx**2+my**2+c1)*(vx+vy+c2)))
        return float(np.mean(out))
    ev=dict(device=torch.cuda.get_device_name(0) if dev=="cuda" else "cpu",
        n_val=len(va), n_params=n_params, lpips_w=LPIPS_W,
        mse_model=mse(pred,Xv), mse_mean_img=mse(mean_img,Xv),
        mse_random=mse(rand_img,Xv), mse_shuffle=mse(pred_shuf,Xv),
        ssim_model=ssim(pred,Xv), ssim_mean=ssim(mean_img,Xv), ssim_random=ssim(rand_img,Xv))
    ev["improve_vs_mean_pct"]=100*(ev["mse_mean_img"]-ev["mse_model"])/ev["mse_mean_img"]
    ev["improve_vs_shuffle_pct"]=100*(ev["mse_shuffle"]-ev["mse_model"])/ev["mse_shuffle"]
    # save a few reconstructions for figure
    np.savez(f"{OUT}/decoder_recon_sample.npz",
        real=Xv[:32].cpu().numpy(), pred=pred[:32].cpu().numpy(),
        mean_img=mean_img[:1].cpu().numpy())
json.dump(ev, open(f"{OUT}/decoder_eval.json","w"), indent=2)
print("EVAL", json.dumps(ev,indent=2), flush=True)
