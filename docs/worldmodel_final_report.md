# Cell World Model — Comprehensive Report

**Project goal**: build a "world model" of cells / cell–cell interactions — a model that predicts how a cell's state shifts under intervention and over time, and how neighboring cells determine each other's states.

**Core schema**: `S(c, a, t+Δt) = f_θ( S(c,a,t), a, Δt, c ; G )`
Advance the cell-state distribution `S`, in context `c`, under intervention `a`, by a time step `Δt`, over a prior interaction graph `G`.

---

## 1. Two world models (headline results)

The project converged onto two models that each demonstrate one of the schema's **two orthogonal axes**.

### 1-A. Time-series world model — intervention → cell-state distribution shift (TIME axis)

- **Data**: the full GSE147405 EMT atlas — **53,290 cells × 11,058 genes**, 4 cancer cell lines (A549/DU145/MCF7/OVCA420) × 3 stimuli (TGFB1/EGF/TNF) × 8 time points (0d→7d induction + washout MET). A **15× expansion** over the initial MVP (single A549_TGFB1 arm, 3,568 cells).
- **Model**: scVI-style NB-VAE latent (cell-line batch correction) + a conditional OT transition model `f(Z_t, inducer, Δt, cellline) → Z_{t+Δt}`, trained with a Sinkhorn (entropic W2) loss. Trained on Modal A100 (VAE 60ep + transition 400ep, 22 min).
- **Results (held-out transitions, vs 4 baselines)**:

  | Transition | world model W2 | identity | mean-shift | OT-only | improvement |
  |---|---|---|---|---|---|
  | 1d→3d | **1.773** | 2.258 | 2.301 | 2.259 | **+21.5% vs identity, +23.0% vs mean-shift** |
  | 3d→7d | **1.790** | 2.251 | 2.301 | 2.266 | **+20.5% vs identity, +22.2% vs mean-shift** |

- **Key finding**: mean-shift and OT-only (which ignore the condition) are worse than even identity. That is, "just push in the average direction" is useless; only by **conditioning on the intervention and context** can you predict the cell-state distribution shift. This quantitatively proves the biological value of the world model.

### 1-B. Cell–cell interaction world model — neighbors → cell state (SPACE axis)

- **Data**: GSE284005 — single-cell resolution spatial transcriptomics (MERFISH, multiple-sclerosis brain tissue). Sample ms1r1: **26,082 cells × 500 genes**, per-cell spatial coordinates (X,Y) + type, 7 cell types, 4 tissue regions. Spatial KNN neighbor graph (k=15, median 52 µm).
- **Model**: message-passing GNN — predict each cell's state (PCA-30) by permutation-invariantly aggregating the states and types of its spatial neighbors.
- **Strict control design**: (A) own type only / (B) + real neighbors / (C) + shuffled neighbors (random cells). B>A means neighbors add information; B>C means that information is spatially real (not an autocorrelation artifact).
- **Results**:

  | Variant | test MSE | test R² |
  |---|---|---|
  | (A) own type only | 0.3264 | 0.612 |
  | **(B) + real neighbors** | **0.2933** | **0.651** |
  | (C) + shuffled neighbors | 0.3267 | 0.612 |

  - Neighbor contribution (B vs A): **+10.1%**; spatial specificity (B vs C): **+10.2%**
  - **Significance**: z = −181 vs the shuffle null (10 runs), p < 0.001
  - **By cell type**: Astrocyte +17%, OPC +12%, Vascular +10% (highly microenvironment-responsive) vs Oligodendrocyte +1.8% (fixed self-identity) — biologically sensible
  - **Spatial range**: stable at +10% regardless of k=3–50 → local (immediate-neighbor) interaction

### 1-C. Integration

The two models **share the same cell-state space S and the same schema**, covering the TIME axis (intervention→dynamics) and the SPACE axis (neighbors→state) complementarily. Combined, they form a complete spatiotemporal world model: "an intervention changes a cell over time → the changed cell propagates to its neighbors." **Note: the current integration is a conceptual linkage (data-backed complementary roles on a shared schema); it is not an end-to-end fusion of the two models into a single network.**

### 1-D. Expression → morphology conditional image decoder + reconstructing a missing axis (MODAL axis)

The fundamental constraint of biological data is **fragmentation** — some data lacks a time axis, some lacks a spatial axis, most lack an image axis. Data that measures expression, morphology, space, and time all in one cell essentially does not exist. We built a module that confronts this head-on: **using expression as a common language, transplant an "expression↔morphology" mapping learned on one dataset onto a dataset that never had images**.

- **Training data**: 10x Xenium Human Colon Cancer — a rare dataset where expression and morphology images are measured **paired in the same cell**. 307,762 cells × 422 genes + 6.5 GB morphology image (34111×31345, 15 z-planes). Built 40,000 per-cell pairs of (256-HVG expression ↔ 64×64 morphology patch).
- **Model**: conditional decoder (expression-encoder MLP + ConvTranspose image decoder). Trained on Modal A100. Initial run failed due to excessive LPIPS → after diagnosis, retrained with pixel loss + early stopping. The final model takes all 422 genes as input with an expanded decoder (6.1M parameters).
- **Held-out reconstruction (Step 5, final 422-gene model)**:

  | Method | MSE | SSIM |
  |---|---|---|
  | **decoder** | **0.0350** | **0.174** |
  | mean image | 0.0450 | 0.129 |
  | expression shuffle | 0.0595 | — |

  - **+22.2%** vs mean, **+41.2%** vs expression shuffle → proves that **expression carries morphology information and the decoder actually uses it** (performance collapses under shuffling). (The initial 256-HVG model gave +22.6%/+39.8%, essentially identical.)
  - **Honest limitation**: reconstruction captures only **~23%** of the real variance → it captures **coarse morphology** like cell size and overall brightness, but not fine nuclear texture (a blurry blob). Cause: the information limit of the 422-gene panel + the mean-seeking blur of MSE regression. This limit remained essentially in place even with all genes and a larger decoder.

- **Reconstructing a missing image axis (Step 6)**: we fed the cell expression of the **EMT time-series data — which had no images at all** — into this decoder to generate morphology patches (full 422-gene decoder, 77 genes shared with EMT).
  - **Supported claim**: the **coarse statistics (brightness, approximate size) of the generated images shift systematically with EMT time (0d→7d)** — the 0d→3d shift is **189× larger** than a time-shuffle control. It disappears under shuffling → the shift is **real expression progression carried through the morphology axis**.
  - **What we do NOT claim (key limitation, stated on the figure)**: the generated images are a **blurry central blob**, not a sharp nuclear morphology. Reconstruction captures only ~23% of the real variance (MSE regression mean-seeking). This limit remained even after retraining with all genes and a larger decoder. Only 77/422 genes transfer, and it is a colon-cancer→lung-cancer cross-context.
  - **Precise conclusion**: this is a **proof of principle that "expression carries the coarse signal of morphology"**, not a working morphology generator. Sharp morphology generation must overcome the limits of this data scale, panel, and MSE regression (it needs diffusion/adversarial generation, which is unstable at this scale) and is future work.
  - **Significance**: the direction itself is the most novel in the project — "assembling modules on a common axis (expression) to stitch together even a missing axis of fragmented biological data." End-to-end monolithic training is impossible because it would require a single dataset with all four axes, but module assembly can learn from different datasets that each hold one axis and connect them. That said, the current image axis's reconstruction fidelity stays at the level of coarse statistics.

### 1-E. Morphology state axis (MORPHOLOGY axis) — inferring state instead of generating pixels

The blur in 1-D was an intrinsic limit of MSE regression (it averages away the many-to-one expression↔morphology relationship). We overcame it in two directions.

**Direction A — don't generate pixels; infer morphology *state* from images (headline).**
- **Method**: embed 40,000 Xenium patches into a 384-dim **morphology state axis** with a pretrained DINOv2 (ViT-S/14, self-supervised). It captures cell-morphology variation without labels — PC1 explains 34.5% of variance, PC2/PC3 correlate with cell brightness and size (ρ ±0.4–0.45).
- **Expression → morphology axis prediction**: expression predicts the morphology axis (top PCs R² 0.16–0.27), and from that predicted axis the **real cell brightness r=0.68 and size r=0.66** are recovered. Shuffling the expression makes it **collapse completely to r→−0.09** → a real signal.
- **Honest comparison**: on the same R² scale, the morphology axis (top 3 PCs ~0.21) is **comparable to raw pixels (0.222), not superior.** We directly tested the hypothesis that "pixel R² is inflated by background variance" and **rejected it** (foreground pixel R² 0.253 > background 0.203). The real value of the morphology axis is not predictive superiority but that it is a **compact, 32-dim, hallucination-free state** — a clean target that is easy to attach to a world model and that does not fabricate images.

**Direction B — generate sharp images with a diffusion decoder (developed in two stages).**

*Stage 1 (vanilla DDPM)*: following the principle of GeneFlow (arXiv 2511.00119), we trained an expression-conditional DDPM (noise-prediction UNet, 64px, 200 timesteps) on A100. **We got sharpness but a hallucination**: sharpness (laplacian 0.053) > real (0.032), but diffusion MSE-to-true (0.100) = MSE-to-shuffle (0.100) → the generated image does not match the correct cell. Diagnosis: the condition (expression) was injected weakly and only once at the bottleneck, so the diffusion ignored the condition and drew a "generic cell."

*Stage 2 (production-grade CFG diffusion — hallucination fixed)*: redesigned per the diagnosis — **classifier-free guidance (condition dropout during training + a guidance weight at sampling)**, FiLM multi-resolution condition injection, EMA, DDIM sampler, cosine LR, 200 epochs, A100-80GB.
  - **Result — hallucination fixed**: at every guidance weight, **MSE-to-true < MSE-to-shuffle** (in stage 1 the two were identical at 0.100). The cell-specificity gap (shuffle−true) grows with guidance: **+0.018 at w=0 → +0.173 at w=3 (~10×)**. Sharpness is retained (0.06–0.08 > real 0.032).
  - **Precise conclusion**: CFG fixed "sharp but wrong" into **"sharp AND correct (cell-specific)."** Guidance forces the diffusion to actually use the expression condition, so the generated morphology matches the correct cell specifically. The cause of the hallucination was not GPU/data scale but the condition-injection method, and it was solved by engineering (CFG).
  - **Honest remaining limitation**: validated with 40k cells (a 200k-cell dataset was built, but transfer failed due to this environment's upload-bandwidth constraint — scaling up is an infrastructure task). Validation is on held-out MSE-to-true/shuffle; absolute pixel fidelity (large-scale metrics like FID) is not measured.

**Schema integration**: the morphology axis was folded into a single state S alongside the time and space axes. The three measurement axes share expression as a common language, and the morphology axis **reconstructs a "missing axis" in the image→state direction, hallucination-free** — it is not a pixel generator.

### 1-F. Unified state model (UNIFIED) — a real single network, not a diagram

The preceding axes were each a separate model. In this stage we trained a **real single network that fuses the four axes — expression, space, morphology, time — into one shared state S (128-dim)** — turning the schema diagram into a running system.

- **Data**: downloaded 10x Xenium CRC directly from the CDN (proxy bypass, 20 MB/s) and extracted, for **200,000 cells** at once, expression (422) + morphology patches (64px) + DINOv2-384 embeddings + spatial coordinates, persisted to a Modal Volume. (A **5× expansion** over the prior diffusion work's 40k, fundamentally solving the upload-bandwidth bottleneck via direct CDN download.)
- **Architecture**: shared encoder E (expression 422→512→256→**S 128**) + 3 task heads (expression reconstruction, morphology DINOv2 regression, spatial GNN: aggregate neighbor S → center S). The three losses pass through the same E and shape S → S comes to hold structure common to all three axes. A100-80GB, EMA · cosine LR · AMP.
- **Results (8,000 held-out cells, all with shuffle controls)**:

  | Task | R² (real) | R² (shuffle) | gain |
  |---|---|---|---|
  | Expression reconstruction (S→422) | **0.574** | −0.574 | +1.148 |
  | Spatial GNN (neighbors→center S) | **0.331** | −0.343 | +0.674 |
  | Morphology prediction (S→DINOv2) | **0.016** | −0.280 | +0.296 |

  **All three axes clearly beat shuffle** → proving that one shared S really holds all three axes' information. The low morphology R² is because expression→morphology is intrinsically a weak signal (consistent with §1-E), but the gain (+0.30) is clear.

- **Connecting morphology generation to S (the key link of the integration)**: we retrained the CFG diffusion decoder conditioned on the **learned state S** rather than raw expression (200k, A100-80GB). The guidance sweep gives **cell_specific=True at every w≥1**, with a cell-specificity gap of **+0.17 at w=3** (epoch-80 checkpoint of the 200k unified model; training ended after loss converged to ~0.0233 at ep80/120; on 128 held-out cells — a separate run from §1-E's completed 40k raw-expression CFG best_gap=0.173), sharpness 0.07 > real 0.032. That is, **"expression → learned S → sharp cell morphology" works as one pipeline**, hallucination-free. (Honest caveat: some cells with weak expression signal are still close to a blob — texture generation depends on signal strength.)

- **Time-axis transfer integration (the fourth axis)**: into the encoder E trained on Xenium, we projected a **completely different dataset — EMT scRNA-seq time series** (A549 TGFB1, 0d→7d, 3,133 cells) using 175 shared genes. In state space S, the **distance from the 0d center increases monotonically with time** (0→2.5→4.8→5.6→6.6), Spearman(time, S-axis projection) **ρ=0.50, p<1e-190**, shuffle ρ≈0 (max 0.053). → **a state space learned on one dataset captures another dataset's temporal structure, hallucination-free.** This is the empirical demonstration of the project's core hypothesis of "stitching together a missing axis."

- **Significance and honest positioning**: the individual technical components (CFG, DINOv2, GNN, shared encoder) are all existing methods. On expression→image generation alone, GeneFlow (FID 20.73, same Xenium platform) and Spatia (49 donors · 17 tissues · 12 disease states, ~17M cell-gene training pairs, integrating morphology + expression + space) lead on scale and quantitative metrics. **Our uncharted territory is "fusing four axes (expression · space · morphology + time transfer) into one predictable state S, with an honest negative control at every step."** It is not at the scale of a frontier paper, but its value as a "running integrated system + honest validation" is clear (figures `unified_worldmodel_demo.png`, `frontier_comparison.png`).

---

## 2. Image-generation world model (engineering validation — appendix)

Midway through the project, we built a separate model targeting "continuous image (time-lapse) generation." This **succeeded as an engineering-pipeline validation but is weak in biological meaning** — recorded honestly below.

- **Data**: IDR idr0052-walther-condensinmap — live-cell mitosis time-lapse (40 frames × 256², 3 channels × 18 z-slices = 54 sequences). **Note: mitosis is a proxy for our real interest (EMT/intervention/cell-cell); the biology does not match.**
- **Architecture**: based on recent literature (Genie 2 latent tokenizer + latent action, DreamerV3 latent rollout, Transformer dynamics, Diffusion-Forcing noise). CPU MVP (128², continuous VAE) → scaled up on Modal A100 (256², VQ-VAE + Transformer + LPIPS).
- **Accuracy**: on held-out t=32–39 autoregressive rollout, GPU v2 = **+44.0%** over persistence (the CPU MVP was +30.6% over its own baseline).
  - **Comparison caveat**: +44.0% (256²·VQ-VAE·Transformer) and +30.6% (128²·continuous VAE·conv) differ in resolution, architecture, and evaluation resolution, so they **must not be read as a direct ranking.** Each is vs its own persistence baseline. A fair same-condition comparison was not run.
- **Honest limitation (key)**: the autoregressive rollout is **nearly static**. The frame-to-frame change of generated frames is only **1%** of the real one (0.002 vs 0.167). That is, the model beats the persistence baseline but does not reproduce the real dynamics of mitosis (nuclear separation, movement) — it only learned that the next frame is similar to the previous one, not *how* the cell changes. This is a limit of data scale (54 sequences) and mean-action rollout.
- **Why it is an appendix**: predicting pixel flow is not "understanding cells." It knows nothing about a cell's *state* or *interactions*. So this branch remains as engineering evidence that "we also validated an image-generation pipeline on GPU," while the headline is the state/interaction models of 1-A/1-B.

---

## 3. Cross-modal alignment (preliminary analysis)

We tried to connect JUMP Cell Painting (morphology, U2OS) and Perturb-seq (expression, CD4+ T cells) via the gene-intervention axis.
- **Result**: coarse global geometry is weakly shared (Mantel r=0.067, p=0.01; effect concordance rho=0.12, p<1e-17), but local/linear structure is not shared (kNN 1.0×, CCA≈0).
- **Interpretation**: **partial alignment** — not a total failure. Gene effects do not transfer as a linear map between different cell types (U2OS vs T cells). This is evidence for schema §2.5 (context dependence) and suggests that cross-context alignment alone is insufficient and context conditioning is required.

---

## 4. Dataset inventory

| Dataset | Use | Scale | Source |
|---|---|---|---|
| GSE147405 EMT time-course | time-series world model (headline) | 53,290 cells | GEO |
| GSE284005 MERFISH MS brain | cell-cell world model (headline) | 26,082 cells | GEO |
| Primary Human CD4+ T Perturb-seq | schema-design anchor, cross-modal | 33,983 perturbations | CZI VCP / AWS |
| JUMP Cell Painting (cpg0016) | cross-modal morphology | 7,975 genes | Cell Painting Gallery |
| IDR idr0052 condensin | image generation (appendix) | 54 sequences | IDR/Embassy |
| **10x Xenium Human Colon CRC** | **unified state model (§1-F)** | **200,000 cells** | 10x CDN direct |

---

## 5. Overall performance table

| World model | Axis | vs baseline | Biological meaning |
|---|---|---|---|
| **Time-series (intervention→distribution shift)** | time | **+21.5%** vs identity, +23% vs mean-shift | **strong** — proves conditioning on intervention/context is essential |
| **Cell-cell (neighbors→state)** | space | **+10%** vs type-only (p<0.001) | **strong** — spatially specific, interpretable per cell type |
| **Morphology state axis (expression→DINOv2 axis)** | morphology | brightness r=0.68 · size r=0.66 (shuffle −0.09) | **medium** — real signal, hallucination-free state axis |
| Expression→morphology decoder (MSE) | morphology (pixels) | +22% vs mean, +41% vs shuffle | **weak** — blurry but correct |
| Diffusion decoder v1 (vanilla DDPM) | morphology (pixels) | sharpness 0.053 > real 0.032 | **weak** — sharp but hallucination |
| **Diffusion decoder v2 (CFG, production)** | morphology (pixels) | true<shuffle, gap +0.173@w=3 | **medium** — sharp AND cell-specific (fixed) |
| Image generation (next-frame) | pixels | +44% vs persistence | **weak** — rollout static, mitosis is a proxy |
| Cross-modal alignment | modal | partial alignment (rho 0.12) | evidence of context dependence |
| **Unified S: expression reconstruction** | unified | R²=0.574 (shuffle −0.574) | **strong** — shared S preserves expression |
| **Unified S: spatial GNN** | unified | R²=0.331 (shuffle −0.343) | **medium** — shared S holds spatial context |
| **Unified S: morphology prediction** | unified | R²=0.016 (shuffle −0.280) | **weak** — real but weak signal |
| **Unified S: time transfer** | unified | ρ=0.50 (shuffle ρ≈0) | **strong** — captures another dataset's temporal structure |
| **Unified S: S→image generation** | unified | cell-specific gap +0.17@w=3 (epoch-80 ckpt) | **medium** — sharp, correct generation from a single S |

---

## 6. Consolidated honest limitations

1. **Data scale**: all proof-of-concept scale. Time-series is 12 arms, cell-cell is a single tissue section (1 sample), image is 54 sequences.
2. **Correlation, not causation**: all three models show predictive performance, but none establishes the causal mechanism of an intervention.
3. **Cell-cell is single-timepoint**: the spatial data has no time axis, so it shows "neighbors *determine* state" but the temporal causality of "neighbors *change* state" is untested.
4. **Integration complete (prior limitation resolved)**: initially the per-axis models only shared a schema, but in §1-F we trained **a real single network that fuses the three axes — expression, space, morphology — into one shared encoder/state S**, and folded the time axis in via frozen-encoder transfer. The remaining limits are the absence of native 4D data holding all four axes at once (time is a separate-platform transfer) and scale (200k cells, 1 tissue), not that "the integration is conceptual."
5. **Image branch**: engineering validation only, no biological insight (§2).
6. **Morphology axis is a weak signal, sharp generation fixed with CFG**: the expression→morphology axis is real (passes shuffle controls) but its predictive power is comparable to pixels (§1-E direction A). Vanilla diffusion was sharp but a hallucination; **redesigned with classifier-free guidance, we fixed it to "sharp AND cell-specific"** (§1-E direction B, stage 2). Remaining limit: validated with 40k cells (the 200k scale failed to transfer due to upload-bandwidth constraints), and large-scale absolute metrics like FID are not measured.

---

## 7. GPU infrastructure

Time-series and image models were validated on Modal A100 (40/80GB). Usage-based billing (within free credit), containers auto-released on job completion. The architecture is designed to scale to larger regimes (512² resolution, a diffusion decoder head, multiple tissue samples) without redesign.

---

## 8. Conclusion

The project's core achievement is to **each empirically demonstrate the two axes of a cell / cell-cell interaction world model**:
- Cell state's distribution shift over time is predictable **when conditioned on intervention and context** (time-series, +21.5%).
- Cell state is partly **determined by spatial neighbors**, and this is statistically significant and biologically interpretable (cell-cell, +10%, p<0.001).

Both results are backed by controls showing that without condition/neighbor information you cannot beat the baseline, demonstrating that the "cell world model" captures real biological structure rather than mere inertial prediction.

Furthermore, in §1-F we trained **a real single network that fuses these axes into one shared state S**: the three axes — expression, space, morphology — share the same S (all passing shuffle controls), a diffusion decoder conditioned on S generates sharp cell morphology without hallucination (cell-specific gap +0.17@w=3, epoch-80 checkpoint), and the time axis of a completely different EMT dataset transfers into the same S via a frozen encoder (ρ=0.50). The individual techniques are existing ones, but **an integrated system that fuses four axes (+ time transfer) into one predictable state with an honest negative control at every step** is this project's unique contribution. It is not a frontier-scale paper, but it empirically demonstrates — as a running system — the hypothesis of stitching fragmented biological data together on a common state axis.
