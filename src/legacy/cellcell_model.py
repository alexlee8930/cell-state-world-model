"""
cell-cell interaction WORLD MODEL — message-passing over a spatial neighbor graph.

Question: does a cell's state depend on the states of its spatial neighbors,
beyond what its own cell type already predicts?

Model: predict center-cell state Z_i from
    - center cell type embedding
    - permutation-invariant aggregation of messages from k spatial neighbors,
      each message = f(neighbor_state, neighbor_type)

Rigorous controls separate true micro-environment signal from spatial
autocorrelation:
    (A) type-only          : no neighbor input
    (B) + real neighbors   : k spatial-KNN neighbors
    (C) + shuffle neighbors: k RANDOM cells (destroys spatial structure)
  B beating A  => neighbors carry information beyond cell type
  B beating C  => that information is spatially real, not a random-cell artifact

Data: GSE284005 (MERFISH, single-cell-resolution spatial atlas). Each cell has
(x,y) coordinates, a 500-gene expression vector (-> PCA-30 state), and a type.

Result (sample ms1r1, 26,082 cells, k=15):
    (A) type only            MSE 0.3264  R² 0.612
    (B) + real neighbors     MSE 0.2933  R² 0.651   (+10.1% vs A)
    (C) + shuffle neighbors  MSE 0.3267  R² 0.612   (B is +10.2% vs C)
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

class CellCellModel(nn.Module):
    def __init__(self, n_types, lat=30, use_neighbors=True):
        super().__init__()
        self.use_neighbors = use_neighbors
        self.type_emb = nn.Embedding(n_types, 16)
        self.msg = nn.Sequential(nn.Linear(lat+16, 64), nn.ReLU(), nn.Linear(64, 64))
        in_dim = 16 + (64 if use_neighbors else 0)
        self.head = nn.Sequential(nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.ReLU(),
                                  nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, lat))
    def forward(self, center_tid, neigh_states, neigh_tid):
        te = self.type_emb(center_tid)
        if self.use_neighbors:
            nte = self.type_emb(neigh_tid)                       # (B,K,16)
            m = self.msg(torch.cat([neigh_states, nte], -1))     # (B,K,64)
            agg = m.mean(1)                                      # perm-invariant
            h = torch.cat([te, agg], -1)
        else:
            h = te
        return self.head(h)
