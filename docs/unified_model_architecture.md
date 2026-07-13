# Unified Cell-State World Model — Architecture

## Core principle
Not a schematic diagram, but a **real single network** that fuses four axes into one learned state space S.
Because a single Xenium CRC dataset contains expression, space, and morphology all at cell-level resolution (200,000 cells),
we jointly train the three axes to be predicted from the same state S. The time axis is folded in by transfer:
an EMT time course is projected into the same S space through the learned encoder.

## Components

### Shared encoder E: expression → state S
- Input: 422-gene expression (CP-median + log1p + z-score)
- MLP: 422 → 512 → 256 → **128 (state S)**, GELU + LayerNorm + residual
- S is the single latent that every task shares. This is what "unified" actually means.

### Multi-task decoder heads (all take the same S as input)
1. **Expression reconstruction** S→422: self-consistency. Forces S to preserve expression information.
2. **Morphology prediction** S→DINOv2-384: regression. Forces S to carry cell morphology (a hallucination-free embedding).
3. **Spatial GNN** aggregate neighbor cells' S → predict the center cell's S: forces S to carry spatial context.
   (k-NN graph, mean-aggregation message passing)
4. **CFG diffusion decoder** S→sharp image (64px): generates a cell-specific image conditioned on S (Step 4).
5. **Temporal trajectory** movement of EMT within S-space: folded in by transfer (Step 5).

### Joint loss
L = λ_recon·MSE(expression) + λ_morph·MSE(DINOv2) + λ_spatial·MSE(neighbors→center S)
- All three losses pass through the same encoder E and shape S → S comes to hold the structure common to the three axes.
- EMA (0.999), cosine LR, AMP, A100-80GB.

## Why this differs from prior work
- GeneFlow / GE2Hist: expression→image generation only. No space, no time.
- Spatia: integrates morphology + expression + space, but static tissue, no time axis.
- **Ours**: fuses expression, space, and morphology into a shared S + **transfers an EMT time axis into that same S** →
  the first attempt to have four axes share one predictable state space. An honest negative control at every step.

## Honest limitations
- The time axis is transfer, not native (EMT is a separate scRNA-seq dataset, a different platform from Xenium).
- Scale is 200,000 cells from a single tissue (short of the millions of cells and multiple tissues of GeneFlow/Spatia).
- Learns correlational, not causal, structure.
