# stereoTrack

**stereoTrack** is a computational framework for spatial trajectory inference (TI) in spatial transcriptomics (ST) data. It integrates gene expression profiles and spatial location information to reconstruct spatiotemporal cell orders at single-cell resolution.

stereoTrack leverages **Masked Autoencoder (MAE)**-based cell embeddings and an **Optimal Transport (OT)**-driven pseudotime framework to capture cellular dynamics within a single tissue section (2D) or across multiple serial sections (3D). It further identifies key transcription factor regulators driving cellular trajectories.

---

## 🌟 Key Features

- **Spatial-aware embedding**: MAE encoder with gene masking and cell masking strategies to learn robust cell and niche representations
- **Optimal Transport pseudotime**: Constructs trajectory by computing OT matrices that jointly consider gene expression distance and spatial transition cost
- **Velocity field**: Estimates cell velocity and visualizes directional dynamics on spatial grids
- **Multi-slice 3D support**: Integrates multiple tissue sections along a Z-axis for 3D spatiotemporal reconstruction
- **Transcription factor analysis**: Identifies molecular regulators governing cellular trajectories

---

## 🔧 Installation

```bash
git clone https://github.com/zEpoch/stereoTrack.git
cd stereoTrack
pip install -e .
```

**Dependencies**: Python ≥ 3.10, PyTorch, DGL, Scanpy, POT (Python Optimal Transport), NumPy, SciPy, scikit-learn

---

## 🚀 Quick Start

### 1. Single-slice (2D) trajectory inference

```python
import scanpy as sc
from stereotrack import run_track

adata = sc.read_h5ad("your_data.h5ad")

# Run full stereoTrack pipeline
run_track(
    adata,
    select_way="coordinates",   # or "cell_type"
    start_point=[-1050, 3850],  # spatial starting point
    data_type="spatial",
    alpha1=0.6,                 # weight for gene expression distance
    alpha2=0.4,                 # weight for spatial distance
    use_rep="cell_embedding",
    n_neigh_pos=50,
)

# Results stored in:
# adata.obs['ptime']      - pseudotime
# adata.uns['E_grid']     - spatial grid positions
# adata.uns['V_grid']     - velocity vectors on grid
```

### 2. Multi-slice (3D) pipeline

```bash
# Step 1: Preprocess
python preprocess.py --config config/config_hanMouse3D_v2.yaml

# Step 2: Train MAE
python train_ddp.py --config config/config_hanMouse3D_v2.yaml

# Step 3: Inference (get cell/niche embeddings)
python inference.py --config config/config_hanMouse3D_v2.yaml --checkpoint out/.../best.pt
```

### 3. Configuration file example

```yaml
paths:
  data_path: /path/to/your/data.h5ad
  save_dir: /path/to/output

preprocess:
  batch_label: z_axis
  spatial_key: ccf

model:
  hidden_dim: 512
  latent_dim: 64
  dropout_rate: 0.1
  gene_mask_ratio: 0.5
  cell_mask_ratio: 0.2

training:
  patch_size: 4096
  n_epochs: 50
  learning_rate: 0.001
  patience: 20
```

---

## 📐 Method Overview

stereoTrack consists of two main stages:

1. **Embedding stage (MAE)**
   - Constructs a spatial adjacency graph via Delaunay triangulation
   - Trains a Masked Autoencoder with two masking strategies:
     - *Gene masking*: randomly masks 50% of genes to learn transcriptomic structure
     - *Cell masking*: randomly masks 20% of cells to learn spatial microenvironment
   - Produces `cell_embedding` (cell-intrinsic) and `niche_embedding` (spatially aggregated)

2. **Trajectory stage (OT + Velocity)**
   - Computes an OT matrix balancing gene expression cost (`alpha1`) and spatial distance cost (`alpha2`)
   - Estimates pseudotime (`ptime`) from user-defined starting cells
   - Computes a velocity field on a spatial grid for visualization

---

## 📁 Repository Structure

```
stereoTrack/
├── stereotrack/
│   ├── mae.py          # MAE encoder (gene & cell masking)
│   ├── model.py        # VAE-based StereoTrackModel (multi-slice)
│   ├── train.py        # Training loop
│   ├── dataset.py      # Graph dataset & dataloader
│   ├── process.py      # Graph construction & preprocessing
│   └── utils.py        # OT matrix, pseudotime, velocity
├── preprocess.py       # Data preprocessing pipeline
├── inference.py        # Model inference script
├── config/             # YAML configuration files
└── README.md
```

---

## 📊 Datasets

stereoTrack has been tested on:
- Mouse brain 3D spatial transcriptomics (Han et al.)
- *(Add your tested datasets here)*

---

## 📝 Citation

If you use stereoTrack in your research, please cite:

```bibtex
@article{stereotrack2025,
  title   = {stereoTrack: Spatiotemporal trajectory inference for spatial transcriptomics},
  author  = {Your Name et al.},
  journal = {TBD},
  year    = {2025}
}
```

---

## 📄 License

MIT License