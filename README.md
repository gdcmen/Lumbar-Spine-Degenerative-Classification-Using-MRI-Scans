# Lumbar Spine Degenerative Classification Using MRI Scans

## Project Overview
- This project aims to develop a Convolutional Neural Network (CNN) deep learning model for analyzing MRI scans of the lumbar spine, specifically focused on three spinal conditions:

    1. Spinal Canal Stenosis
    2. Neural Foraminal Narrowing
    3. Subarticular Stenosis
- The objective is to accurately classify each of these conditions as Normal, Moderate, or Severe based on MRI images. This project involves a multi-condition, multi-output CNN model that takes a single MRI image as input and outputs the state of each condition.

- To enhance the model’s predictive capabilities, we explore different MRI orientations (Axial and Sagittal) and implement advanced multi-output architectures tailored for CNNs to optimize predictions for each condition.

*Note: This repository reflects ongoing work, and only the current progress is shown. The project is still under development, and updates will be made as the work progresses. This is not the final product.*

### Example Image
Below is an example of a preprocessed MRI image, showing annotated regions relevant to the classification of spinal conditions.

![76438935_2212277614_13](https://github.com/user-attachments/assets/6fb4b47d-5797-4174-82f2-7dcb5c8a015e)

*Figure: Annotated MRI scan indicating areas associated with spinal conditions. The model classifies these regions as Moderate, or Severe based on the Left Subarticular Stenosis at the L3-L4 level*
## Project Goals and Requirements
1. Single-Input, Multi-Condition Output Design
  - Our primary aim is to create a model that can analyze a single MRI scan and provide predictions for each of the three conditions. Each condition will have a separate output, representing its specific state (Normal, Moderate, or Severe). This design involves:

    - A shared feature extraction stage where the model learns common features from the input MRI image.
    - Separate prediction heads for each condition, allowing for condition-specific state predictions.
    - This architecture ensures that the model efficiently utilizes shared features while maintaining distinct outputs tailored to each condition.

2. Axial and Sagittal MRI Models
  - The project also investigates the impact of MRI orientation on model performance. Three variations of the model are explored:

    - `Model 1`: Trained on Axial MRI images only.
    - `Model 2`: Trained on Sagittal MRI images only.
    - `Model 3`: Trained on a combination of Axial and Sagittal images.
  - By comparing these models, we aim to determine which orientation, or combination of orientations, provides the most accurate multi-condition predictions.

3. Multi-Output Architecture and Consistency Regularization
  - Given the multi-output nature of the model, we are incorporating advanced multi-output architectures and exploring consistency regularization techniques. These techniques are intended to improve prediction accuracy by enforcing coherence between outputs. This ensures that each condition benefits from both shared and unique features learned within the model.

## Skills and Technologies
This project demonstrates proficiency in the following skills and technologies:

1. `Deep Learning & CNNs`: Building and training Convolutional Neural Networks for medical image analysis, specifically focusing on multi-condition, multi-output architectures.
2. `Medical Imaging`: Processing and analyzing MRI images, including DICOM-to-PNG conversion and region-of-interest annotation for lumbar spine conditions.
3. `Data Preprocessing`: Cleaning, structuring, and organizing large medical datasets, including labeling conditions and levels.
4. `Multi-Output Models`: Designing and implementing multi-output architectures to predict multiple conditions from a single input, optimizing shared and condition-specific features.
5. `Advanced Model Techniques`: Exploring consistency regularization and multi-output architectures to improve model coherence and accuracy across conditions.
6. `Python & Libraries`: Leveraging Python and core libraries such as `TensorFlow/Keras` for modeling, `Pydicom` for `DICOM` handling, `Pillow` for image manipulation, and `Pandas` for data management.
7. `Data Visualization`: Visualizing MRI annotations and model predictions to evaluate and refine model performance.
8. `Project Organization & Documentation`: Structuring a complex project, creating a reproducible workflow, and documenting progress for collaborative and educational purposes.


## Data Structure
For this project, data is organized into three main directories to facilitate preprocessing, labeling, and training. Here’s an overview of how the data is structured:

1. Raw MRI Data
- Directory: `data/raw/`
- Description: Contains the original DICOM files for both Axial and Sagittal MRI scans. These raw files are converted and labeled in the subsequent preprocessing stages.
2. Processed Data
- Directory: `data/processed/`
- Description: After preprocessing, DICOM images are converted to PNG format and saved with labels indicating the specific spinal condition and intervertebral level. Each image file is named according to a pattern that includes identifiers like study_id, series_id, and instance_number to facilitate tracking and model training.
3. Labels
- Directory: `data/labels/`
- Description: Label files include information on the condition state for each intervertebral level, such as Normal, Moderate, or Severe. These labels are used during model training to provide ground truth for each condition.

* Example of Directory and File Naming Convention
    - Processed images and label files are named and organized by the following convention:

    - Image Naming: `<study_id>_<series_id>_<instance_number>.png`
    - Label Structure: Labels are mapped to each image file based on its study and series identifiers, allowing the model to identify specific conditions and levels accurately.
This structure allows for efficient preprocessing, tracking, and model training while keeping the data organized.
