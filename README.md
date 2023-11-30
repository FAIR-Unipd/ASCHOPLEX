# ASCHOPLEX: Automatic Segmentation of Choroid Plexus toolbox

## What is ASCHOPLEX
ASCHOPLEX is a Deep Learning based toolbox for the Automatic Segmentation of Choroid Plexus starting from brain MRI. 

ASCHOPLEX was trained in a 5-folds cross-validation fashion on 128 subjects derived from two different dataset with following characteristics:

    - Dataset 1: 
        - 67 subjects (24 controls, 43 Relapsing-Remitting Multiple Sclerosis)
        - Scanner: Philips Achieva TX with 8-channels head coil (Software version R3.2.3.2)
        - Sequence: 3D T1-w MPRAGE, resolution 1x1x1 mm, SENSE acceleration factor 2.5, TE/TR 3.7/8.4 ms, FA: 9°
    - Dataset 2: 
        - 61 subjects Relapsing-Remitting Multiple Sclerosis
        - Scanner: Philips Elition S with 32-channels head coil (Software version R5.7.2.1)
        - Sequence: 3D T1-w MPRAGE, resolution 1x1x1 mm, SENSE acceleration factor 4, TE/TR 3.7/8.4 ms, FA: 8°

ASCHOPLEX returns the ensemble by major voting segmentation of five selected models (best one for each fold) predictions, improving robustness and reliability of the output segmentation.

ASCHOPLEX was tested on an unseen dataset composed by 77 subjects (26 controls, 51 depressed) acquired on a GE SIGNA PET/MRI scanner with a 3D T1-w Fast SPGR sequence, resolution 1x1x1 mm, TE/TR 2.99/6.96 ms, FA 12°.

Two main modality has been tested:
1. Direct Inference: ASCHOPLEX has been directly inferred on the unseen dataset. The final prediction is the ensemble by major voting of the predictions obtained with the best five-folds models trained on Dataset 1 and Dataset 2.
2. Finetuning: ASCHOPLEX has been finetuned on the unseed dataset varying the number of subjects (1-10) in the training set. Only the previously selected five models has been finetuned using 1-fold cross-validation. The final segmentation derived from the ensemble by major voting of predictions derived by the five finetuned models.

Due to the data-driven nature of ASCHOPLEX, results confirmed the addiction of a finetuning step allows reaching the same performance of the training phase on an unseen dataset. The selected number of subjects in the training set of the finetuning step is 5 (and 5 for the validation).

## How to run ASCHOPLEX

ASCHOPLEX has been structured as a Python tool with selected mandatory and optional input parsers.

    python launching_tool.py --dataroot {path} --work_dir {path} --finetuning {flag} --prediction {flag} --{Optional Parser}

Mandatory Inputs:

    --dataroot: data directory
    --work_dir: working directory
    --finetuning: yes/no, finetuning flag
    --prediction: yes/no/ft, prediction flag

Optional Inputs:

    --description: data description (will be inserted in the JSON file)
    --output_pred_dir: Working directory where to save predictions. If not specified, default folder name and location (work_dir) will be used
    --finetune_dir: Working directory where to save finetuned models. If not specified, default folder name and location (work_dir) will be used

You can selected one of these combinations:

    - Direct Inference: not recommended instead of data match that used for the training of ASCHOPLEX (Dataset 1, Dataset 2). The output segmentation is the ensemble by major voting of the predictions obtained by the best five selected models (trained on Dataset 1 and Dataset 2).

            python launching_tool.py --dataroot {path} --work_dir {path} --finetuning no --prediction yes

    - Finetuning & Prediction: recommended modality. After the finetuning of ASCHOPLEX on 10 manually segmented subjects (5 for training and 5 for validation), the ensemble by major voting is performed using the five finetuned models.

            python launching_tool.py --dataroot {path} --work_dir {path} --finetuning yes --prediction yes

    - Only Finetuning: recommended modality. The finetuning of ASCHOPLEX on 10 manually segmented subjects (5 for training and 5 for validation) has been performed without obtaining the ensemble prediction.

            python launching_tool.py --dataroot {path} --work_dir {path} --finetuning yes --prediction no

    - Prediction after Finetuning: recommended modality for prediction when new data, with same characteristics of the previous ones, are available. The output segmentation is the ensemble by major voting performed using the predictions derived by the five previously finetuned models.

            python launching_tool.py --dataroot {path} --work_dir {path} --finetuning no --prediction ft

## ASCHOPLEX tool steps

ASCHOPLEX pipeline automatically run the following steps basing on --finetuning and --prediction flags:

1. Writing JSON file: A JSON file of the input data will always be written. The JSON file will be saved in {work_dir}/JSON_file folder.
Based on --finetuning and --prediction flags, different files will be created:

        - {work_dir}/JSON_file/dataset_prediction.json for Direct Inference or Prediction after Finetuning
        - {work_dir}/JSON_file/dataset_finetuning_prediction.json for Finetuning & prediction
        - {work_dir}/JSON_file/dataset_finetuning.json for Only Finetuning 

2. (Optional) Finetuning on input dataset: If --finetuning yes, run the finetuning step. It returns the models and the folder path where models have been saved.

3. (Optional) Predictions: If --prediction yes/ft, run the ensemble by major voting. It returns the predictions and the folder path where models have been saved.

## Input Data: Structure and Characteristics

Data must be in Nifti format (.nii, .nii.gz).
Data must be untouched: no preprocessing (e.g. brain extraction, intensity correction) is needed. Images should not be flipped.
Input data must be structured as follows:

    DATASET
        |-images_Tr
            |-MRI_IDsj_image.nii.gz
            |- ....
        |-images_Ts
            |-MRI_IDsj_image.nii.gz
            |- ....
        |-label_Tr
            |-MRI_IDsj_seg.nii.gz
            |- ....

where IDsj must be a unique identifier for the subjects, images_Tr is the folder with 10 subjects images (*image.nii*) with relative manual segmentations (*seg.nii*) saved in label_Tr (necessary only if --finetuning yes), images_Ts is the folder with testing subjects (*image.nii*).
