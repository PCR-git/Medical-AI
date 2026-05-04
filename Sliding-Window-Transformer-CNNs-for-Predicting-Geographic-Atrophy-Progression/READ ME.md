SWAU-Net: Longitudinal GA Prediction

Implementation of SWAU-Net, a hybrid CNN-Transformer for forecasting Geographic Atrophy (GA) expansion in AMD.

Repository Map
models/: SWAU-Net, SWA core, and DynNet architectures.

training_utils/ & eval_utils/: Hybrid loss logic and k-fold validation.

data_utils/ & augmentation_utils/: Sequence loading and FAF noise simulation.

Blob Growth Pretraining.ipynb: Synthetic data generation and Phase 1 training.

GA_proj_MAIN.ipynb: Clinical fine-tuning and main evaluation pipeline.

Highlights
Sliding Window Attention (SWA): Uses temporal weight-sharing to prevent overfitting in low-data regimes.

Decoupled Dynamics: Separates state estimation from growth evolution for cleaner forecasts.

Performance: Achieves 0.66 Growth Mask DSC, significantly outperforming standard Transformers.

Quick Start
Pretrain: Run Blob Growth Pretraining.ipynb for anisotropic growth simulation.

Fine-tune: Use GA_proj_MAIN.ipynb for clinical 5-fold cross-validation.
