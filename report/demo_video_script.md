# 3-Minute Demo Video Script
## "One State to Draw Them All — A Unified Cell-State World Model"
**Presenter:** Yuchan Lee · **Event:** Built with Claude — Life Sciences Hackathon (Researcher Track)

Judging weights this script is built around: **Impact 25% · Claude Use 25% · Depth & Execution 20% · Demo 30%.**
Total runtime ≈ 3:00. Timings are cumulative. Narration is written to be read aloud verbatim; *italics* are on-screen actions.

---

### [0:00–0:25] — The hook: biology is fragmented (Impact)
*On screen: four separate panels drift in — a gene-expression heatmap, a microscope image, a spatial tissue slide, a time-lapse strip — each in its own box, not touching.*

> "Biology has always measured a cell one window at a time. A transcriptome in one experiment. A microscope image in another. A spatial slide in a third. A time-lapse in a fourth — different platforms, different distributions, and **no shared coordinate system**. Nobody had asked them to become one thing.
> Our thesis is that they already *are* one thing. A cell is a dynamical system with a single hidden state — its regulatory program — and expression, shape, position, and time are just four windows onto that one state."

---

### [0:25–0:50] — The idea: one state, four axes (Impact + Depth)
*On screen: the four boxes collapse into a single glowing node labeled **State S (128-d)**; four arrows fan out from it to the four axes.*

> "So we forced all four to share **one** 128-dimensional latent state S — and made S a proper *world model*: something you can **read** by decoding it to any axis, **roll forward** through time, **intervene** on, and **decode back into an actual cell image**.
> To our knowledge, this is the **first cell model to hold all four axes in one intervenable, generative state.** Almost the entire virtual-cell field today — Arc's STATE, CZI's TranscriptFormer, scGPT — is single-modality transcriptomics predicting perturbation-to-expression. None of them draws the cell, places it in tissue, and moves it through time from one shared state."

---

### [0:50–1:35] — LIVE DEMO (Demo — the heaviest-weighted section)
*On screen: switch to the browser, real URL visible in the address bar:*
**`https://alexlee--cell-world-model-demo-web.modal.run`**

> "This is live and public — anyone can open it right now. It's serving our actual trained models on a GPU."

*Action: the page auto-loads a grid. Top row = real Xenium cells, bottom row = images generated purely from S. Point at a pair.*

> "Top row: real cells from the Xenium colon-cancer dataset. Bottom row: the model has **never seen these images**. It only saw each cell's 422-gene expression, encoded it into state S, and a classifier-free-guidance diffusion decoder **drew the morphology from S alone.**"

*Action: change the seed to pick new held-out cells; click Generate. New pairs appear in ~2 seconds.*

> "New held-out cells, generated in about two seconds."

*Action: drag the guidance-weight slider up; click Generate again.*

> "The guidance slider is the intervention knob — turning it up sharpens how strongly the image commits to that cell's state. This is the axis-reconstruction idea made real: give the model an axis a dataset never captured, and it fills it in."

---

### [1:35–2:15] — Depth: we earned this the hard way (Depth & Execution)
*On screen: the `sharp_vs_blob_comparison.png` figure — REAL / MSE-blurry / diffusion-sharp rows.*

> "This wasn't a straight line, and the honest path is the point. We first built the standard world-model recipe — autoregressive frames plus a diffusion decoder, like Genie 2 — on live-cell time-lapse. It beat the baseline by 41%, and **we rejected it as a success**, because it had only learned the optical flow of pixels and understood nothing about the cell.
> Then a plain decoder gave us correct-but-blurry blobs. Then diffusion gave us razor-sharp images that were **hallucinations** — they fit the wrong cell just as well as the right one. Our *sharpest* output was our *least* faithful."

*On screen: the frontier_comparison bar panel — every axis vs. its shuffle control.*

> "So we built a negative control into **every single axis**: a cell-specificity gap, expression R² versus shuffle, a shuffled-time control. Nothing is asserted; everything beats its shuffle. Knowing when a generated axis is *real* versus *fabricated* is, to us, part of the contribution."

---

### [2:15–2:45] — The bridge: time transfer (Impact + Depth)
*On screen: the `unified_time_axis.png` trajectory — S-distance rising 0→2.5→4.8→5.6→6.6 over EMT time.*

> "And the state generalizes across platforms. We froze the encoder trained on static Xenium and, with **no retraining**, projected a completely separate EMT time-course from a different platform into the same S. The state moved monotonically with real biological time — correlation 0.5, p below 1e-190, while a shuffled control sits at zero.
> That's the mechanism: a relationship learned on one assay transferring to another. It's how fragmented single-cell data — siloed today by platform — could be stitched into one navigable space, and how you'd run in-silico perturbations before touching a wet lab."

---

### [2:45–3:00] — Close (Impact + Claude Use)
*On screen: the hero figure (unified_worldmodel_demo.png), then the GitHub repo + live URL + paper title card.*

> "One shared state. Four axes. A control at every step. Trained end-to-end on a single A100, served live and open-source.
> This is a first step toward a virtual cell you can actually *read, draw, and steer* — not just a schematic, but a running model. Everything — code, the technical paper, and the live demo — is public. Thank you."

*End card: `github repo · alexlee--cell-world-model-demo-web.modal.run · Yuchan Lee`*

---

## Production notes
- **Screen-capture the live demo section in one continuous take** — the 2-second generation is the single most convincing moment; don't cut it.
- If the first demo request is slow, it is a scale-to-zero cold start (~30–60 s). **Warm it once right before recording** by loading the page, then record.
- B-roll to have ready: the six figures in `results/figures/` (architecture, hero, sharp-vs-blob, training, gallery, time-axis, frontier).
- Keep the tone confident but honest — the "we rejected our own success" beat is what separates this from a slide deck.
- On-screen lower-thirds for the three numbers judges remember: **ρ = 0.50 (time transfer)**, **cell-specificity gap +0.17**, **first 4-axis intervenable cell state**.
