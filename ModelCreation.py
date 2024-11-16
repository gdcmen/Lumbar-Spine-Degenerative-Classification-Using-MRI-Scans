# Load the Drive helper and mount
from google.colab import drive

drive.mount('/content/drive')

import pandas as pd
import numpy as np
import tensorflow as tf
import os
import pydicom
import pydicom.data
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw
from tqdm import tqdm
from sklearn.cluster import KMeans
import joblib
from sklearn.model_selection import train_test_split



# Directories
train_labels_path = "/content/drive/MyDrive/Colab Notebooks/Senior Project/rsna-2024-lumbar-spine-degenerative-classification/train.csv"
train_series_description_path = "/content/drive/MyDrive/Colab Notebooks/Senior Project/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv"
train_labels_coordinates = "/content/drive/MyDrive/Colab Notebooks/Senior Project/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv"

dicom_dir = "/content/drive/MyDrive/Colab Notebooks/Senior Project/rsna-2024-lumbar-spine-degenerative-classification/dicom_images"
output_dir_png = "/content/drive/MyDrive/Colab Notebooks/Senior Project/rsna-2024-lumbar-spine-degenerative-classification/png_images"

"""# Process Raw Data
Import the new images, organize them and include them into the data that is being used.
- First we organize the data in the dataset, so it can be iterated by the algorithm later in the code
"""

df_series_description = pd.read_csv(train_series_description_path)
df_train_labels = pd.read_csv(train_labels_path)

print(df_series_description.shape)
df_series_description.tail()
#The type of data in study id is integer

# Coordinates of the trained labels

train_coordinates = pd.read_csv(train_labels_coordinates)
print(train_coordinates.shape)
df_coords = train_coordinates
train_coordinates.head(n=10)

development_dataset = pd.merge(df_series_description, df_train_labels, on="study_id")
print(development_dataset.shape)
development_dataset.head(n=10)

# Define a mapping for the categorical labels to numerical labels
mapping = {'Normal/Mild': 1, 'Moderate': 2, 'Severe': 3}

# Replace all occurrences in the dataset
development_dataset.replace(mapping, inplace=True)

# Choose only the images from the Axial type of MRIs for the Axial model
# Filter the data to include only axial images
df_summary = development_dataset[development_dataset['series_description'].str.contains("Axial", case=False)]

print(df_summary.shape)
df_summary.head()

print("Columns in df_summary:")
print(df_summary.columns.tolist())

"""## Create a dataset that will be passed into the model with the pictures"""

# Get a list of all files in the folder
file_list = os.listdir(dicom_dir)
img_path = random.choice(file_list)
dcm_data = pydicom.dcmread(dicom_dir + "/" + img_path, force=True)
print(dcm_data.SeriesInstanceUID)
dcm_data

"""Calculate the location of the MRI's based on their slice location, and iterating through a big sample of dcm files

* Due to the high volume of data, we take a sample of 500 rows to calculate from not on.
"""

# Parameters
num_files_to_process = 5000
sample_size = 5000  # Number of rows to sample for slice_df
spinal_levels = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']

# Gather slice locations
slice_locations = []
all_dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
random_files = random.sample(all_dicom_files, min(num_files_to_process, len(all_dicom_files)))

for f in tqdm(random_files, desc="Gathering Slice Locations"):
    try:
        dcm = pydicom.dcmread(os.path.join(dicom_dir, f))
        if 'SliceLocation' in dcm:
            slice_locations.append((f, float(dcm.SliceLocation), dcm.InstanceNumber))
    except Exception as e:
        print(f"Error reading file {f}: {e}")

# Convert to DataFrame
slice_df = pd.DataFrame(slice_locations, columns=['filename', 'slice_location', 'instance_number'])

# Sample 500 rows from slice_df
slice_df = slice_df.sample(n=sample_size, random_state=1)
print("Sampled slice_df created with 500 rows.")

slice_df.shape

slice_df.tail()

"""We keep the KMeans from the full dataset, since that would give us better classification on the unlabaled data"""

# Clustering with KMeans for 5 clusters
slice_locations_array = slice_df['slice_location'].values.reshape(-1, 1)
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(slice_locations_array)

# Sort cluster centers by slice location to align with spinal levels
cluster_centers = sorted(kmeans.cluster_centers_.flatten())
slice_df['level'] = slice_df['slice_location'].apply(
    lambda loc: spinal_levels[np.argmin([abs(loc - center) for center in cluster_centers])]
)
slice_df['cluster'] = kmeans.predict(slice_locations_array)

# Verify assignment
print(slice_df[['filename', 'slice_location', 'instance_number', 'level']].head())

import seaborn as sns
import matplotlib.pyplot as plt

# Density plot of slice locations by cluster
plt.figure(figsize=(12, 8))
for cluster in range(5):
    cluster_data = slice_df[slice_df['cluster'] == cluster]
    if not cluster_data.empty:  # Check if there's data in this cluster
        sns.kdeplot(cluster_data['slice_location'], label=f'Cluster {cluster}', fill=True)

# Adding labels and title
plt.xlabel("Slice Location")
plt.title("Density Plot of Slice Locations for Each Cluster")
plt.legend(title="Clusters")
plt.show()

# Box plot of slice locations by cluster
plt.figure(figsize=(12, 8))
sns.boxplot(data=slice_df, x='cluster', y='slice_location', palette="Set3")
plt.xlabel('Cluster')
plt.ylabel('Slice Location')
plt.title('Box Plot of Slice Locations by Cluster')
plt.show()



# Save the KMeans model to an .h5 file using joblib
joblib.dump(kmeans, '/content/drive/MyDrive/Colab Notebooks/Senior Project/rsna-2024-lumbar-spine-degenerative-classification/kmeans_model.joblib')
print("Model saved as 'kmeans_model.joblib'")

# Save the slice data
slice_df.to_csv('/content/drive/MyDrive/Colab Notebooks/Senior Project/rsna-2024-lumbar-spine-degenerative-classification/slice_data.csv', index=False)
print("Slice data saved as 'slice_data.csv'")

# Load the KMeans model
kmeans = joblib.load('/content/drive/MyDrive/Colab Notebooks/Senior Project/rsna-2024-lumbar-spine-degenerative-classification/kmeans_model.joblib')  # Use .joblib if that’s the format used to save the model
print("Model loaded from 'kmeans_model.joblib'")

# Load the slice data
slice_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Senior Project/rsna-2024-lumbar-spine-degenerative-classification/slice_data.csv')
print("Model and slice data loaded.")

slice_df.head()

slice_df['level'].value_counts()

slice_df['filename'] = slice_df['filename'].str.replace('.dcm', '', regex=False)
# Split 'filename' to create 'study_id', 'series_id', and 'instance' columns
slice_df[['study_id', 'series_id']] = slice_df['filename'].str.split('_', n=2, expand=True)[[0, 1]]
# Convert 'study_id' and 'series_id' to integer types for consistency
slice_df['study_id'] = slice_df['study_id'].astype(int)
slice_df['series_id'] = slice_df['series_id'].astype(int)

level = slice_df['level'].str.replace('/', '_').str.lower()

# Convert the sets to lists to enable indexing.
filename = list(slice_df['filename'])
study_id_list = list(slice_df['study_id'])
slice_locations = list(slice_df['slice_location'])
series_id_list = list(slice_df['series_id'])
instance_list = list(slice_df['instance_number'].astype(str)) # Convert to strings
level_list = list(slice_df['level'].str.replace('/', '_').str.lower()) # Convert to strings


print("df_summary columns:", df_summary.columns)


# Access the first element of the lists
print("File name:", filename[0],
      "\nSlice location:", slice_locations[0],
      "\nStudy ID:", study_id_list[0],
      "\nSeries ID:", series_id_list[0],
      "\nInstance:", instance_list[0],
      "\nLevel:", level_list[0])
print(slice_df.columns)

"""Code to check what is the structure of the data prior to fixing together both of the datasets"""

# Ensure data types are consistent between slice_df and df_summary
slice_df['study_id'] = slice_df['study_id'].astype(int)
slice_df['series_id'] = slice_df['series_id'].astype(int)
df_summary['study_id'] = df_summary['study_id'].astype(int)
df_summary['series_id'] = df_summary['series_id'].astype(int)

# Enhanced safe_get function with detailed debug output
def safe_get(df, study_id, series_id, level, condition, default=np.nan):
    column_name = f"{condition}_{level}"  # Column name based on level and condition
    if column_name in df.columns:
        result = df.loc[
            (df['study_id'] == study_id) & (df['series_id'] == series_id),
            column_name
        ]
        # Detailed debug prints
        print(f"Searching for: {column_name} in df_summary with study_id={study_id} and series_id={series_id}")
        print("Match found:", not result.empty, "Result:", result.values if not result.empty else "No match")
        return result.values[0] if not result.empty else default
    print(f"Column {column_name} not found in df_summary")
    return default

# Testing each entry in slice_df
for _, row in slice_df.iterrows():
    study_id = row['study_id']
    series_id = row['series_id']
    instance_number = row['instance_number']
    level = row['level']

    # Test each grade retrieval with enhanced safe_get debug output
    spinal_canal_stenosis_grade = safe_get(df_summary, study_id, series_id, level, 'spinal_canal_stenosis')
    left_neural_foraminal_narrowing_grade = safe_get(df_summary, study_id, series_id, level, 'left_neural_foraminal_narrowing')
    right_neural_foraminal_narrowing_grade = safe_get(df_summary, study_id, series_id, level, 'right_neural_foraminal_narrowing')
    left_subarticular_stenosis_grade = safe_get(df_summary, study_id, series_id, level, 'left_subarticular_stenosis')
    right_subarticular_stenosis_grade = safe_get(df_summary, study_id, series_id, level, 'right_subarticular_stenosis')


# Assuming slice_df and df_summary are already loaded with the specified structure

# Define columns for final_df
columns_names = [
    "study_id",
    "series_id",
    "instance_number",
    "level",
    "spinal_canal_stenosis_grade",
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis"
]

final_df = pd.DataFrame(columns=columns_names)

# Standardize `level` naming in `slice_df` to match df_summary's format (e.g., "L1/L2" to "l1_l2")
slice_df['level'] = slice_df['level'].str.replace('/', '_').str.lower()

# Iterate through each row in `slice_df`
for _, row in slice_df.iterrows():
    study_id = row['study_id']
    series_id = row['series_id']
    instance_number = row['instance_number']
    level = row['level']

    # Create a helper function to retrieve values safely from `df_summary`
    def safe_get(df, study_id, series_id, level, condition, default=np.nan):
        column_name = f"{condition}_{level}"  # Target column based on level and condition
        if column_name in df.columns:
            result = df.loc[
                (df['study_id'] == study_id) & (df['series_id'] == series_id),
                column_name
            ]
            return result.values[0] if not result.empty else default
        return default

    # Retrieve grades for each condition
    spinal_canal_stenosis_grade = safe_get(df_summary, study_id, series_id, level, 'spinal_canal_stenosis')
    left_neural_foraminal_narrowing_grade = safe_get(df_summary, study_id, series_id, level, 'left_neural_foraminal_narrowing')
    right_neural_foraminal_narrowing_grade = safe_get(df_summary, study_id, series_id, level, 'right_neural_foraminal_narrowing')
    left_subarticular_stenosis_grade = safe_get(df_summary, study_id, series_id, level, 'left_subarticular_stenosis')
    right_subarticular_stenosis_grade = safe_get(df_summary, study_id, series_id, level, 'right_subarticular_stenosis')

    # Create a new row for final_df
    new_row = pd.DataFrame({
        "study_id": [study_id],
        "series_id": [series_id],
        "instance_number": [instance_number],
        "level": [level],
        "spinal_canal_stenosis_grade": [spinal_canal_stenosis_grade],
        "left_neural_foraminal_narrowing": [left_neural_foraminal_narrowing_grade],
        "right_neural_foraminal_narrowing": [right_neural_foraminal_narrowing_grade],
        "left_subarticular_stenosis": [left_subarticular_stenosis_grade],
        "right_subarticular_stenosis": [right_subarticular_stenosis_grade]
    })

    # Append the new row to final_df
    final_df = pd.concat([final_df, new_row], ignore_index=True)

# Display the first few rows of final_df to confirm
print(final_df.head())

# Save final dataset to CSV
output_path = '/content/drive/MyDrive/Colab Notebooks/Senior Project/rsna-2024-lumbar-spine-degenerative-classification/Final_MRI_Dataset.csv'
final_df.to_csv(output_path, index=False)
print(f"Final dataset saved to {output_path}")

# Remove rows with NaN values in any of the specified grading columns
grading_columns = [
    "spinal_canal_stenosis_grade",
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis"
]

# Drop rows with NaN values in any of the grading columns
final_df_cleaned = final_df.dropna(subset=grading_columns)

# Display the cleaned DataFrame to confirm NaN rows are removed
print(final_df_cleaned.head())

# Save the cleaned dataset to CSV
output_path = '/content/drive/MyDrive/Colab Notebooks/Senior Project/rsna-2024-lumbar-spine-degenerative-classification/Final_MRI_Dataset_Cleaned.csv'
final_df_cleaned.to_csv(output_path, index=False)
print(f"Cleaned dataset saved to {output_path}")

"""# Organize images

As we have +2000 images, we will store them into 5 different folders, one for each of the conditions, so we can train the model on the different conditions. When we create the model, we will structure it to have 5 different outputs.
"""

# Load cleaned dataset with conditions
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Senior Project/rsna-2024-lumbar-spine-degenerative-classification/Final_MRI_Dataset_Cleaned.csv')
print("Column names in df:", df.columns)

# Define paths
dicom_dir = r'/content/drive/My Drive/Colab Notebooks/Senior Project/rsna-2024-lumbar-spine-degenerative-classification/dicom_images'
output_dir = r'/content/drive/My Drive/Colab Notebooks/Senior Project/rsna-2024-lumbar-spine-degenerative-classification/processed_images'

# Create directories for each condition if they don't exist
conditions = [
    "spinal_canal_stenosis",
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis"
]
for condition in conditions:
    os.makedirs(os.path.join(output_dir, condition, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, condition, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, condition, 'test'), exist_ok=True)

# Function to convert DICOM to PNG
def convert_dicom_to_png(dicom_path, png_path):
    try:
        dcm = pydicom.dcmread(dicom_path)
        img_array = dcm.pixel_array
        plt.imsave(png_path, img_array, cmap='gray')
    except Exception as e:
        print(f"Error converting {dicom_path}: {e}")

def process_images(df, dicom_dir, output_dir):
    # Define a mapping of conditions to their respective column names in df
    condition_columns = {
        "spinal_canal_stenosis": "spinal_canal_stenosis_grade",
        "left_neural_foraminal_narrowing": "left_neural_foraminal_narrowing",
        "right_neural_foraminal_narrowing": "right_neural_foraminal_narrowing",
        "left_subarticular_stenosis": "left_subarticular_stenosis",
        "right_subarticular_stenosis": "right_subarticular_stenosis"
    }

    for _, row in df.iterrows():
        study_id = str(row['study_id'])
        series_id = str(row['series_id'])
        instance_number = str(row['instance_number'])

        # Locate DICOM file based on `filename` convention
        dicom_filename = f"{study_id}_{series_id}_{instance_number}.dcm"
        dicom_path = os.path.join(dicom_dir, dicom_filename)

        # Only proceed if the DICOM file exists
        if os.path.exists(dicom_path):
            # Convert to PNG
            png_filename = f"{study_id}_{series_id}_{instance_number}.png"
            png_path = os.path.join(output_dir, png_filename)
            convert_dicom_to_png(dicom_path, png_path)

            # Organize images into folders based on conditions
            for condition, column_name in condition_columns.items():
                # Check if the column exists in df and has a non-NaN value
                if column_name in df.columns and not pd.isna(row[column_name]):
                    # Assign to train, validate, or test set
                    condition_folder = os.path.join(output_dir, condition)
                    train_val_test_split(condition_folder, png_path, condition, row[column_name])


# Function to split images between train, validation, and test sets
def train_val_test_split(condition_folder, png_path, condition, label):
    # Determine destination directory
    if np.random.rand() < 0.8:
        split = 'train'
    elif np.random.rand() < 0.9:
        split = 'val'
    else:
        split = 'test'

    # Define final path for this condition and split
    destination = os.path.join(condition_folder, split, f"{label}_{os.path.basename(png_path)}")
    Image.open(png_path).save(destination)
    print(f"Saved {png_path} to {destination}")

# Run the processing
process_images(df, dicom_dir, output_dir)

print("Images processed succesfuly!")
