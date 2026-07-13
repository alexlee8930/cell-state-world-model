"""Public Gradio demo for the Unified Cell-State World Model.
Deploy on Modal:  modal deploy serve_app.py  → persistent public URL (scale-to-zero).
Loads the frozen encoder E + S-conditioned CFG diffusion decoder from the
xenium-unified Volume, lets anyone pick a held-out cell and generate its
morphology from the learned state S with an adjustable guidance weight.
"""
import modal

app = modal.App("cell-world-model-demo")
vol = modal.Volume.from_name("xenium-unified")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.3.1", "numpy", "gradio==4.44.0", "pillow"))

# ---------------- model definitions (must match training) ----------------
MODEL_SRC = '''
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, math
class Encoder(nn.Module):
    def __init__(self,e,s):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(e,512),nn.GELU(),nn.LayerNorm(512),
            nn.Linear(512,256),nn.GELU(),nn.LayerNorm(256),nn.Linear(256,s))
    def forward(self,x): return self.net(x)
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
        self.pool=nn.AvgPool2d(2); self.cdim=cdim
    def temb_fn(self,t):
        half=128; f=torch.exp(torch.arange(half,device=t.device)*-(math.log(10000)/(half-1)))
        a=t[:,None].float()*f[None]; return torch.cat([a.sin(),a.cos()],-1)
    def forward(self,x,t,s,drop=None):
        c=self.temb(self.temb_fn(t))+self.semb(s)
        if drop is not None: c=torch.where(drop[:,None],self.null[None],c)
        h0=self.inp(x); h1=self.d1(h0,c); h2=self.d2(self.pool(h1),c); h3=self.d3(self.pool(h2),c)
        m=self.mid(self.pool(h3),c)
        u=F.interpolate(m,scale_factor=2); u=self.u3(torch.cat([u,h3],1),c)
        u=F.interpolate(u,scale_factor=2); u=self.u2(torch.cat([u,h2],1),c)
        u=F.interpolate(u,scale_factor=2); u=self.u1(torch.cat([u,h1],1),c)
        return self.out(u)
'''
@app.function(image=image, volumes={"/data": vol}, gpu="A10G",
              scaledown_window=300, timeout=600, min_containers=0)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    import torch, numpy as np, gradio as gr, math
    from PIL import Image
    dev="cuda" if torch.cuda.is_available() else "cpu"
    exec(MODEL_SRC, globals())

    # --- load data + models ---
    d=np.load("/data/xenium_unified.npz")
    expr=torch.tensor(np.asarray(d["expr"],dtype=np.float32))
    patches=torch.tensor(np.asarray(d["patches_u8"],dtype=np.float32)/255.0)
    val_idx=np.asarray(d["val_idx"]); val_idx=val_idx[:4000]
    G=expr.shape[1]

    enc=Encoder(G,128).to(dev).eval()
    enc.load_state_dict(torch.load("/data/unified_state_model.pt",map_location=dev)["enc"])
    net=SUNet(128).to(dev).eval()
    ck=torch.load("/data/diffusion_S_ckpt.pt",map_location=dev)
    if isinstance(ck,dict) and "ema" in ck: ck=ck["ema"]
    elif isinstance(ck,dict) and "net" in ck: ck=ck["net"]
    net.load_state_dict(ck)

    T=400
    def cosine_betas(T,s=0.008):
        st=torch.linspace(0,T,T+1); f=torch.cos(((st/T)+s)/(1+s)*math.pi/2)**2
        ac=f/f[0]; b=1-(ac[1:]/ac[:-1]); return b.clamp(1e-4,0.999)
    betas=cosine_betas(T).to(dev); acp=torch.cumprod(1-betas,0)

    @torch.no_grad()
    def ddim(s, w, steps=50):
        n=s.shape[0]; x=torch.randn(n,1,64,64,device=dev)
        ts=torch.linspace(T-1,0,steps).long().to(dev)
        for i,t in enumerate(ts):
            tb=torch.full((n,),int(t),device=dev)
            e_c=net(x,tb,s); e_u=net(x,tb,s,drop=torch.ones(n,dtype=torch.bool,device=dev))
            e=e_u+w*(e_c-e_u)
            a=acp[t]; a_p=acp[ts[i+1]] if i+1<len(ts) else torch.tensor(1.0,device=dev)
            x0=((x-(1-a).sqrt()*e)/a.sqrt()).clamp(-1,1)
            x=a_p.sqrt()*x0+(1-a_p).sqrt()*e
        return x.clamp(-1,1)

    def to_img(arr):
        a=((arr+1)/2).clip(0,1) if arr.min()<0 else arr.clip(0,1)
        return Image.fromarray((a*255).astype(np.uint8)).resize((256,256),Image.NEAREST)

    def generate(seed, w):
        rng=np.random.default_rng(int(seed))
        pick=rng.choice(val_idx, 4, replace=False)
        s=enc(expr[pick].to(dev))
        gen=ddim(s, float(w)).cpu().numpy()[:,0]
        reals=[to_img(patches[p].numpy()) for p in pick]
        gens=[to_img(g) for g in gen]
        # tile real-over-gen pairs
        out=[]
        for r,g in zip(reals,gens):
            canvas=Image.new("L",(256,528),255)
            canvas.paste(r,(0,0)); canvas.paste(g,(0,272))
            out.append(canvas)
        info=(f"Cells {pick.tolist()} · guidance w={w}\n"
              f"Top row = REAL Xenium morphology · Bottom row = GENERATED from state S\n"
              f"Model: frozen encoder (expr→S 128d) + CFG diffusion decoder (epoch-80).")
        return out, info

    with gr.Blocks(title="Cell-State World Model") as demo:
        gr.Markdown("# One State to Draw Them All — Unified Cell-State World Model\n"
                    "Pick a held-out cell; the model encodes its 422-gene expression into a shared "
                    "state **S**, then a CFG diffusion decoder **generates** its morphology from S. "
                    "Top = real Xenium image, bottom = generated. Public demo (Yuchan Lee).")
        with gr.Row():
            seed=gr.Number(value=0, label="Random seed (cell selection)", precision=0)
            w=gr.Slider(0,5,value=3,step=0.5,label="CFG guidance weight w")
            btn=gr.Button("Generate", variant="primary")
        gallery=gr.Gallery(label="Real (top) vs Generated (bottom)", columns=4, height=560)
        info=gr.Textbox(label="Details", lines=3)
        btn.click(generate, [seed,w], [gallery,info])
        demo.load(generate, [seed,w], [gallery,info])
    from fastapi import FastAPI
    fastapi_app=FastAPI()
    return gr.mount_gradio_app(fastapi_app, demo, path="/")
