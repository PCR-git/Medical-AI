# Model-Distillation-for-Segmentation-of-the-Foveal-Avascular-Zone

Code for:  
**Automated Foveal Avascular Zone OCTA Segmentation in Multiple Eye Diseases Using Knowledge Distillation** :contentReference[oaicite:0]{index=0}

---

## Overview

Implementation of a hybrid CNN–Transformer model for **FAZ segmentation in OCTA images**, using **knowledge distillation** to improve performance across multiple retinal diseases.

---

## Repository Structure

- `Construct Dataset/` – Data preprocessing and dataset construction from PNG images  
- `utils/` – Helper functions  
- `FAZ_Multitask_Model_v3.ipynb` – Model implementation (single + multi-task)  
- `FAZ_Multitask_Model_v3_cross_validation.ipynb` – Main training + cross-validation pipeline  

---

## Notes

- Latest version: `FAZ_Multitask_Model_v3_cross_validation.ipynb`  
- Single-task and multi-task models are defined in the same notebook and trained separately  

---

## Quick Start

1. Build dataset using `Construct Dataset/`  
2. Run `FAZ_Multitask_Model_v3_cross_validation.ipynb` for training and evaluation  
