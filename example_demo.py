"""
StereoTrack Demo - Complete workflow example

This script demonstrates how to:
1. Load and preprocess spatial transcriptomics data
2. Train the StereoTrack model
3. Perform inference to get embeddings and imputation
"""

import scanpy as sc
import logging
from pathlib import Path
import pandas as pd

from stereotrack.single_time import (
    train_stereotrack,
    inference_stereotrack,
    seed_all,
    preprocess_adata_with_multislice,
    construct_graph,
    preprocess_adj_sparse,
    get_spatial_input
)


def main():
    # ========== Configuration ==========
    # Set random seed for reproducibility
    seed_all(42)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Paths
    data_path = "./data/spatial_data.h5ad"  # 修改为你的数据路径
    output_dir = "./output"
    
    # Model parameters
    hidden_dim = 512
    latent_dim = 64
    dropout_rate = 0.2
    
    # Training parameters
    n_epochs = 50
    batch_size = 64
    learning_rate = 0.001
    patience = 5
    
    logging.info("=" * 50)
    logging.info("StereoTrack Demo Started")
    logging.info("=" * 50)
    
    # ========== Step 1: Load Data ==========
    logging.info("\n[Step 1/4] Loading spatial transcriptomics data...")
    adata = sc.read_h5ad(data_path)
    logging.info(f"Data shape: {adata.shape}")
    logging.info(f"Number of cells: {adata.n_obs}")
    logging.info(f"Number of genes: {adata.n_vars}")
    
    # ========== Step 2: Preprocess Data ==========
    logging.info("\n[Step 2/4] Preprocessing data...")
    # Standardize gene names to uppercase
    adata.var.index = pd.Index([g.upper() for g in adata.var.index])
    
    # Preprocess expression data
    preprocess_adata_with_multislice([adata])
    logging.info("✓ Expression data preprocessed")
    
    # Construct spatial graph
    construct_graph([adata])
    logging.info("✓ Spatial graph constructed")
    
    # Preprocess adjacency matrix
    preprocess_adj_sparse([adata])
    logging.info("✓ Adjacency matrix normalized")
    
    # Get spatial input features
    get_spatial_input([adata])
    logging.info("✓ Spatial features prepared")
    
    # ========== Step 3: Train Model ==========
    logging.info("\n[Step 3/4] Training StereoTrack model...")
    
    model, adata_trained = train_stereotrack(
        adata=adata,
        save_dir=output_dir,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout_rate=dropout_rate,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience
    )
    
    logging.info("✓ Model training completed")
    logging.info(f"✓ Training results saved to {output_dir}")
    
    # ========== Step 4: Inference ==========
    logging.info("\n[Step 4/4] Performing inference...")
    
    model_path = f"{output_dir}/trained_model/stereotrack_best.pt"
    
    adata_result = inference_stereotrack(
        adata=adata_trained,
        model_path=model_path,
        save_dir=output_dir,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout_rate=dropout_rate,
        batch_size=batch_size
    )
    
    logging.info("✓ Inference completed")
    logging.info(f"✓ Results saved to {output_dir}")
    
    # ========== Summary ==========
    logging.info("\n" + "=" * 50)
    logging.info("StereoTrack Demo Completed Successfully!")
    logging.info("=" * 50)
    logging.info("\nOutput files:")
    logging.info(f"  - Trained model: {output_dir}/trained_model/stereotrack_best.pt")
    logging.info(f"  - Training result: {output_dir}/stereotrack_result.h5ad")
    logging.info(f"  - Inference result: {output_dir}/stereotrack_inference_result.h5ad")
    logging.info(f"  - Embeddings: {output_dir}/stereotrack_embeddings.npy")
    logging.info(f"  - Imputation: {output_dir}/stereotrack_imputation.npy")
    logging.info("\nYou can access:")
    logging.info("  - Embeddings: adata.obsm['stereotrack_embedding']")
    logging.info("  - Imputation: adata.layers['stereotrack_imputation']")


if __name__ == "__main__":
    main()
