Code for paper: Automated Foveal Avascular Zone OCTA Segmentation in Multiple Eye Diseases Using Knowledge Distillation

The most updated version of the code is FAZ_Multitask_Model_v3_cross_validation, which also includes code to perform cross validations of the trained models.
Single-task and mult-task models are both defined in this single code. They were trained separately by running first one part of the code, then the other.
It's not the cleanest way to do things, but at least it's all in one place.

Files in Construct Dataset were used to preprocess and build data from PNGS.
Helper functions are in utils.
