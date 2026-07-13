# One State to Draw Them All — A Unified Cell-State World Model

**Author:** Yuchan Lee
**Event:** Built with Claude — Life Sciences Hackathon (Researcher Track)
**License:** MIT

### 📄 [**Read the full technical report (PDF)**](report/unified_paper_full.pdf)

Single-cell biology is fragmented: expression, morphology, spatial context, and
time are measured on different platforms, with no shared coordinate system. This
project builds a **single trained network** in which one 128-dimensional latent
state **S** is shared across all four axes — and validates every axis against an
honest shuffle / identity control.

**Why it matters.** A cell is a dynamical system with one hidden state (its
regulatory program); expression, morphology, position, and time are just four
windows onto it. We treat `S` as a proper *world model* — one you can **read**
(decode to any axis), **roll forward** in time, **intervene on**, and **decode
back into an image**. To our knowledge this is the **first cell model to hold
all four axes in one intervenable, generative state**. Almost the entire
virtual-cell field today (Arc STATE 167M cells, CZI TranscriptFormer 112M, scGPT,
GeneFlow, Spatia) is single-modality transcriptomics predicting
perturbation→expression; none is a four-axis world model. The individual blocks
(CFG, DINOv2, GNN, FiLM) are adapted from known work — the novelty is the fusion,
and proving each axis *real* with an adversarial negative control rather than
asserting it. Our sharpest diffusion images turned out to be hallucinations that
fit the wrong cell as well as the right one, so we shipped the coarse-but-correct
reconstruction instead: **knowing when a generated axis is real vs. fabricated is
part of the contribution.**

<p align="center">
  <img src="results/figures/unified_worldmodel_demo.png" width="90%">
</p>

## 🔴 Live public demo

Try the model in your browser — no install required:

### **https://alexlee--cell-world-model-demo-web.modal.run**

Pick a held-out cell; the frozen encoder maps its 422-gene expression into the
shared state **S**, then a classifier-free-guidance diffusion decoder
**generates** the cell's morphology from S. Top row = real Xenium image,
bottom row = generated. Adjust the guidance weight `w` to trade contrast for
condition-fidelity. Served on Modal (A10G, scale-to-zero — the first request
after idle takes ~30–60 s to cold-start, then it is instant).

To redeploy your own instance: `modal deploy serve_app.py`.

---

## Headline result

On **200,000 cells** from a 10x Xenium human colon-cancer sample, a shared encoder
maps expression → S, and multiple heads decode S back to each axis. **Every axis
beats a shuffled control on held-out cells:**

| Axis | Metric | Real | Shuffle control |
|------|--------|------|-----------------|
| Expression reconstruction | held-out R² | **0.574** | −0.574 |
| Spatial (neighbor→center via GNN) | held-out R² | **0.331** | −0.343 |
| Morphology (S→DINOv2 embedding) | held-out R² | **0.016** | −0.280 (gain +0.30) |
| Time transfer (EMT → S projection) | Spearman ρ | **0.50** | ~0 (max\|ρ\|=0.05) |
| S → image generation | cell-specific gap @w=3 | **+0.17** | 0 |

A classifier-free-guidance (CFG) diffusion decoder generates **sharp, cell-specific**
images directly from S (sharpness 0.074 > real 0.032, no hallucination), and the
**same frozen encoder** transfers an *independent* EMT time-course (GSE147405, A549
TGFB1, 0d→7d) into S, where cells trace a monotone trajectory.

---

## Repository layout

```
cell-state-world-model/
├── README.md                     # this file
├── LICENSE                       # MIT
├── requirements.txt              # Python dependencies
├── report/
│   ├── unified_paper_full.pdf    # full technical paper (LaTeX) — the main report
│   ├── unified_paper_full.tex    #   └ LaTeX source
│   ├── hackathon_summary.txt     # short submission summary
│   └── demo_video_script.md      # 3-minute demo video script
├── src/
│   ├── 01_data_prep_unified.py   # Xenium → (expr, patches, DINOv2, coords) on a Modal Volume
│   ├── 02_train_unified.py       # shared encoder E + 3 heads, joint training (A100)
│   ├── 03_train_diffusion_S.py   # S-conditioned CFG diffusion decoder
│   ├── 04_sample_diffusion_S.py  # guidance sweep + sampling / evaluation
│   ├── embed_dinov2.py           # self-supervised morphology embeddings
│   └── legacy/                   # earlier single-axis world-model components
├── results/
│   ├── unified_all_metrics.json  # all headline numbers
│   ├── unified_train_log.json    # per-epoch joint-training loss
│   ├── unified_time_axis.json    # time-transfer stats
│   ├── unified_eval.json         # held-out evaluation
│   └── figures/                  # the report figures (all English)
└── docs/
    ├── worldmodel_final_report.md      # comprehensive technical write-up
    └── unified_model_architecture.md   # architecture design notes
```

---

## Pipeline (how to reproduce)

The four numbered scripts in `src/` are the unified pipeline, in order. They are
written to run on a single **A100-80GB** GPU (we used [Modal](https://modal.com));
the data step downloads directly from the 10x CDN (~20 MB/s) into a persistent
Volume, bypassing upload bottlenecks.

1. **`01_data_prep_unified.py`** — range-downloads three members from the Xenium
   `_outs.zip` (cell-feature matrix, cells table, morphology OME-TIFF), extracts
   64×64 DAPI patches + 422-gene expression + (x,y) centroids + DINOv2-384
   embeddings for 200k cells, and writes `xenium_unified.npz` to the Volume.
2. **`02_train_unified.py`** — trains the shared encoder (422→512→256→128) with
   three heads (expression recon, morphology, spatial GNN) jointly; saves
   `unified_state_model.pt` and precomputes the latent `S` for all cells.
3. **`03_train_diffusion_S.py`** — freezes the encoder, precomputes S, and trains
   a CFG DDPM UNet **conditioned on S (128-d), not raw expression**, via FiLM at
   every resolution. Orphan-safe: commits an EMA checkpoint + progress JSON to the
   Volume every few epochs.
4. **`04_sample_diffusion_S.py`** — loads the latest checkpoint, runs a guidance
   sweep (w = 0…5), and reports the cell-specificity gap (MSE-to-true vs
   MSE-to-shuffle) plus sharpness.

The **time axis** is a transfer: the frozen encoder from step 2 projects an
independent EMT scRNA-seq time-course (175 genes shared with the Xenium panel)
into the same S; the monotone trajectory (ρ=0.50) is computed in
`unified_time_axis.json`.

---

## Data (all public)

- **Spatial / morphology / expression / space:** 10x Genomics Xenium
  `Xenium_V1_Human_Colon_Cancer_P1_CRC_Add_on_FFPE` (307,762 cells, 422-gene panel,
  6.5 GB morphology image). We use a 200k-cell subset.
- **Time axis (transfer):** GEO **GSE147405** — A549 TGFB1-induced EMT time-course
  (0d, 8h, 1d, 3d, 7d), 3,133 main-arm cells.

---

## Honest limitations

We state these plainly in the report. (1) Scale: 200k cells, one tissue — below
frontier scale. (2) Morphology is a genuinely weak signal (many-to-one
expression→shape); some generated cells remain blob-like. (3) Correlation, not
causation. (4) Time is a *transfer*, not native 4D data (EMT is a separate
platform). (5) No absolute image metric (e.g. FID) — we report a cell-specificity
gap instead. The final diffusion run stopped at epoch 80/120 after the loss had
converged (~0.0233); all figures use the epoch-80 checkpoint.

Every building block (CFG, DINOv2, GNNs, a shared encoder) is established. The
contribution here is the **combination** — four axes fused into one predictable
state S, with a negative control at every step — built end-to-end during the event
with Claude Science.

## Related work

On expression→image generation alone, **GeneFlow** (2025, FID 20.73, same Xenium
platform) and **Spatia** (2025; 49 donors, 17 tissue types, 12 disease states,
~17M cell-gene training pairs; fuses morphology + expression + space) exceed us in
scale and absolute metrics. Our unexplored niche is the four-axis fusion with an
honest control at every step.
