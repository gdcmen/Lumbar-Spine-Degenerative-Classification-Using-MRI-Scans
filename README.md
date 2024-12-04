# Lumbar Spine Degenerative Classification Using MRI Scans

## Project Overview
- This project focuses on the application of deep learning techniques to diagnose degenerative conditions of the lumbar spine from MRI scans. Currently, the project is dedicated to the classification of Spinal Canal Stenosis, with a 82% accuracy, with future iterations planned to include Neural Foraminal Narrowing and Subarticular Stenosis. Using a Convolutional Neural Network (CNN) architecture, the goal is to provide a reliable, automated diagnostic tool that can aid medical professionals in identifying and classifying spinal conditions.

*Note: This repository reflects ongoing work, and only the current progress is shown. The project is still under development, and updates will be made as the work progresses. This is not the final product.*

### Example Image
Below is an example of a preprocessed MRI image, showing annotated regions relevant to the classification of spinal conditions.

![76438935_2212277614_13](https://github.com/user-attachments/assets/6fb4b47d-5797-4174-82f2-7dcb5c8a015e)

*Figure: Annotated MRI scan indicating areas associated with spinal conditions. The model classifies these regions as Moderate, or Severe based on the Left Subarticular Stenosis at the L3-L4 level*

## Performance
- Current Accuracy: 82% on validation data, using Axial T2 images exclusively.
- Evaluation Metrics:
    - Classification reports highlight precision, recall, and F1 scores for each class.
    - Confusion matrix visualizations assess the model's ability to distinguish between categories.

## Why this Matters

- *Healthcare Relevance*: Lumbar spine conditions are a leading cause of chronic pain and disability worldwide, often requiring MRI-based evaluation for accurate diagnosis. Automating this process can reduce the diagnostic burden on radiologists, speed up treatment decisions, and improve patient outcomes.
- *Advancing Medical Imaging*: This project demonstrates how artificial intelligence can augment medical imaging techniques, delivering consistent and accurate results that complement clinical expertise.

## Current Focus: Spinal Canal Stenosis

- *Objective*: To classify Spinal Canal Stenosis as either Normal, Moderate, or Severe using Axial T2-weighted MRI images.
- Why Axial T2 Images?
    - Axial T2-weighted images are particularly effective in visualizing the spinal canal and adjacent soft tissues, making them ideal for assessing Spinal Canal Stenosis.
    - These images provide high contrast between cerebrospinal fluid and spinal cord, which helps in identifying narrowing or abnormalities in the spinal canal.
 
### Achievements

- A CNN model has been developed exclusively using Axial T2 images, achieving 82% accuracy in classification.
- Preprocessing and data augmentation techniques have been tailored to Axial T2 images to ensure robustness across variations in imaging quality and anatomy.
- The model has demonstrated its ability to generalize to unseen images, accurately predicting the severity of stenosis on non-augmented datasets.

### Example Prediction
- The model has been successfully tested with unseen, non-augmented Axial T2 images, confirming its reliability in real-world scenarios.

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


## Model Architecture
The CNN architecture is composed of:
1. Five Convolutional Blocks:
    - Each block includes convolutional layers, batch normalization, and max pooling.
    - Dropout layers are added for regularization.
2. Fully Connected Layers:
    -  These layers aggregate features and perform the final classification into Normal, Moderate, or Severe categories.
3. Training Configuration:
    - Loss function: sparse_categorical_crossentropy
    - Optimizer: adam
    - Early stopping to prevent overfitting during training.


## Data and Preprocessing
The dataset consists of Axial T2-weighted MRI images processed for consistency and quality:

1. Preprocessing:
    - DICOM images are converted to PNG format.
    - Non-standard or corrupted files are removed.
    - Images are resized to 256x256 pixels for uniformity.
2. Augmentation:
    . To address class imbalance, the following augmentations were applied:
    . Random width/height shifts
    . Zooming
    - Horizontal flipping
- Minor rotations
These augmentations were specifically designed to maintain the integrity of Axial T2 images while improving generalization.


## Project Value and Vision
This project aims to bring meaningful advancements to the field of medical imaging:

- `Enhancing Diagnostics`: By automating MRI analysis, the model provides consistent and accurate second opinions, reducing the workload for radiologists.
- `Improving Access`: With its scalability, this project can extend advanced diagnostic capabilities to remote and underserved areas.
- `Driving AI Integration in Healthcare`: Demonstrating how AI can complement human expertise, this project paves the way for future innovations in medical diagnostics.
