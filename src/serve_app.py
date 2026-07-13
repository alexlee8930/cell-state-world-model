"""Public FastAPI demo for the Unified Cell-State World Model.
Deploy on Modal:  modal deploy serve_app.py  -> persistent public URL (scale-to-zero).

A pure FastAPI app (no Gradio): it loads the frozen encoder E + the S-conditioned
CFG diffusion decoder from the xenium-unified Volume, serves a self-contained HTML
front-end at "/", and exposes GET /api/generate which picks held-out cells,
encodes their 422-gene expression into the shared state S, and generates their
morphology from S at an adjustable guidance weight.
"""
import modal

app = modal.App("cell-world-model-demo")
vol = modal.Volume.from_name("xenium-unified")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.3.1", "numpy", "pillow", "fastapi[standard]"))

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

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>One State to Draw Them All</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--border:#30363d;--text:#d8dee4;--muted:#8b949e;--accent:#388bfd;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
header{border-bottom:1px solid var(--border);padding:14px 20px}
h1{margin:0;font-size:16px}h1 small{color:var(--muted);font-weight:400;font-size:12px}
.sub{color:var(--muted);font-size:12px;margin-top:4px}
main{max-width:900px;margin:0 auto;padding:20px}
.ctrl{display:flex;gap:16px;align-items:center;flex-wrap:wrap;background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:14px}
.ctrl label{font-size:12px;color:var(--muted);display:flex;flex-direction:column;gap:4px}
input[type=number]{width:90px;background:#0d1117;color:var(--text);border:1px solid var(--border);border-radius:4px;padding:6px}
input[type=range]{width:180px}
button{background:var(--accent);color:#fff;border:0;border-radius:4px;padding:8px 18px;font-weight:600;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px}
.cell{background:var(--panel);border:1px solid var(--border);border-radius:6px;overflow:hidden;text-align:center}
.cell img{width:100%;display:block;image-rendering:pixelated}
.cell .lab{font-size:11px;color:var(--muted);padding:4px}
.info{font-family:ui-monospace,monospace;font-size:11px;color:var(--muted);white-space:pre-wrap;margin-top:14px;border-top:1px solid var(--border);padding-top:10px}
.wv{font-family:ui-monospace,monospace;color:var(--text)}
</style></head><body>
<header>
  <h1>One State to Draw Them All &nbsp;<small>Unified Cell-State World Model · Yuchan Lee</small></h1>
  <div class="sub">Pick a held-out cell → the frozen encoder maps its 422-gene expression into the shared state <b>S</b> → a classifier-free-guidance diffusion decoder generates its morphology from S. Top = real Xenium, bottom = generated.</div>
</header>
<main>
  <div class="ctrl">
    <label>Random seed (cell selection)<input id="seed" type="number" value="0" step="1"></label>
    <label>CFG guidance weight w = <span id="wv" class="wv">3.0</span><input id="w" type="range" min="0" max="5" step="0.5" value="3"></label>
    <button id="go">Generate</button>
  </div>
  <div id="grid" class="grid"></div>
  <div id="info" class="info">Loading model (first request after idle cold-starts in ~30-60s)…</div>
</main>
<script>
const wEl=document.getElementById('w'),wv=document.getElementById('wv');
wEl.oninput=()=>wv.textContent=(+wEl.value).toFixed(1);
async function gen(){
  const btn=document.getElementById('go');btn.disabled=true;
  document.getElementById('info').textContent='Generating… (running 50 DDIM steps on GPU)';
  try{
    const r=await fetch(`/api/generate?seed=${document.getElementById('seed').value}&w=${wEl.value}`);
    const d=await r.json();
    document.getElementById('grid').innerHTML=d.images.map((src,i)=>
      `<div class="cell"><img src="${src}"><div class="lab">cell ${d.cells[i]}<br>real / generated</div></div>`).join('');
    document.getElementById('info').textContent=d.info;
  }catch(e){document.getElementById('info').textContent='Error: '+e;}
  btn.disabled=false;
}
document.getElementById('go').onclick=gen; gen();
</script></body></html>"""


@app.function(image=image, volumes={"/data": vol}, gpu="A10G",
              scaledown_window=300, timeout=600, min_containers=0)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    import io, base64, math
    import torch, numpy as np
    from PIL import Image
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    exec(MODEL_SRC, globals())

    # --- load data + models (once per container) ---
    d = np.load("/data/xenium_unified.npz")
    expr = torch.tensor(np.asarray(d["expr"], dtype=np.float32))
    patches = torch.tensor(np.asarray(d["patches_u8"], dtype=np.float32) / 255.0)
    val_idx = np.asarray(d["val_idx"])[:4000]
    G = expr.shape[1]

    enc = Encoder(G, 128).to(dev).eval()
    enc.load_state_dict(torch.load("/data/unified_state_model.pt", map_location=dev)["enc"])
    net = SUNet(128).to(dev).eval()
    ck = torch.load("/data/diffusion_S_ckpt.pt", map_location=dev)
    if isinstance(ck, dict) and "ema" in ck: ck = ck["ema"]
    elif isinstance(ck, dict) and "net" in ck: ck = ck["net"]
    net.load_state_dict(ck)

    T = 400
    def cosine_betas(T, s=0.008):
        st = torch.linspace(0, T, T + 1); f = torch.cos(((st / T) + s) / (1 + s) * math.pi / 2) ** 2
        ac = f / f[0]; b = 1 - (ac[1:] / ac[:-1]); return b.clamp(1e-4, 0.999)
    betas = cosine_betas(T).to(dev); acp = torch.cumprod(1 - betas, 0)

    @torch.no_grad()
    def ddim(s, w, steps=50):
        n = s.shape[0]; x = torch.randn(n, 1, 64, 64, device=dev)
        ts = torch.linspace(T - 1, 0, steps).long().to(dev)
        for i, t in enumerate(ts):
            tb = torch.full((n,), int(t), device=dev)
            e_c = net(x, tb, s); e_u = net(x, tb, s, drop=torch.ones(n, dtype=torch.bool, device=dev))
            e = e_u + w * (e_c - e_u)
            a = acp[t]; a_p = acp[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
            x0 = ((x - (1 - a).sqrt() * e) / a.sqrt()).clamp(-1, 1)
            x = a_p.sqrt() * x0 + (1 - a_p).sqrt() * e
        return x.clamp(-1, 1)

    def to_pil(arr):
        a = ((arr + 1) / 2).clip(0, 1) if arr.min() < 0 else arr.clip(0, 1)
        return Image.fromarray((a * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST)

    def data_uri(img):
        buf = io.BytesIO(); img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def generate(seed, w):
        rng = np.random.default_rng(int(seed))
        pick = rng.choice(val_idx, 4, replace=False)
        s = enc(expr[pick].to(dev))
        gen = ddim(s, float(w)).cpu().numpy()[:, 0]
        uris = []
        for p, g in zip(pick, gen):
            canvas = Image.new("L", (256, 528), 255)
            canvas.paste(to_pil(patches[p].numpy()), (0, 0))
            canvas.paste(to_pil(g), (0, 272))
            uris.append(data_uri(canvas))
        info = (f"cells {pick.tolist()} · guidance w={w}\n"
                f"top = REAL Xenium morphology · bottom = GENERATED from state S\n"
                f"model: frozen encoder (expr→S 128d) + CFG diffusion decoder (epoch-80)")
        return uris, pick.tolist(), info

    api = FastAPI(title="Cell-State World Model")

    @api.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @api.get("/api/generate")
    def api_generate(seed: int = 0, w: float = 3.0):
        images, cells, info = generate(seed, w)
        return JSONResponse({"images": images, "cells": cells, "info": info})

    return api
