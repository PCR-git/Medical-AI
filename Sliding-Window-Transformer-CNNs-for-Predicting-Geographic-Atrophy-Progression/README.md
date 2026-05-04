# SWAU-Net: Longitudinal GA Prediction

Implementation of **SWAU-Net**, a hybrid CNN–Transformer architecture for forecasting **Geographic Atrophy (GA)** expansion in Age-related Macular Degeneration (AMD).

---

## Repository Structure

- `models/`  
  SWAU-Net, Sliding Window Attention (SWA) core modules, and DynNet architectures.

- `training_utils/` & `eval_utils/`  
  Hybrid loss functions and k-fold cross-validation pipeline.

- `data_utils/` & `augmentation_utils/`  
  Sequence loading utilities and FAF noise simulation.

- `Blob Growth Pretraining.ipynb`  
  Synthetic data generation and Phase 1 pretraining.

- `GA_proj_MAIN.ipynb`  
  Clinical fine-tuning and main evaluation pipeline.

---

## Highlights

- **Sliding Window Attention (SWA)**  
  Temporal weight-sharing mechanism that reduces overfitting in low-data regimes.

- **Decoupled Dynamics**  
  Separates latent state estimation from growth evolution for more stable and interpretable forecasts.

- **Performance**  
  Achieves **0.66 Growth Mask DSC**, outperforming standard Transformer baselines.

---

## Quick Start

### 1. Pretraining
Blob Growth Pretraining.ipynb

### 2. Fine tuning and evaluation
GA_proj_MAIN.ipynb
