import sqlite3
from sklearn.datasets import fetch_lfw_people
import cv2
import os
import pickle

# Load the data using sklearn
lfw_dataset = fetch_lfw_people(min_faces_per_person=5, download_if_missing=True, color=True)

db_path = './Model/Datasets/retrain_dataset.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Filter out people with more than 10 images
selected_indices = [i for i in range(len(lfw_dataset.target_names)) if (lfw_dataset.target == i).sum() < 10]
selected_people = [lfw_dataset.target_names[i] for i in selected_indices]

# Select the first 7 people from the filtered list
selected_people = selected_people[:7]

# Filter the dataset to include only the selected people
selected_indices = [i for i in range(len(lfw_dataset.target)) if lfw_dataset.target_names[lfw_dataset.target[i]] in selected_people]
lfw_dataset.data = lfw_dataset.data[selected_indices]
lfw_dataset.images = lfw_dataset.images[selected_indices]
lfw_dataset.target = lfw_dataset.target[selected_indices]

# Dictionary to store the count of images for each target
target_image_count = {}

max_images_per_target = 9  # Maximum number of images per target
max_people = 8  # Maximum number of people in the dataset

# Inserting the targets, names, and images into the table
for image_index, images in enumerate(lfw_dataset.images):
    # Convert the image data to bytes
    rgb_image = (images * 255).astype('uint8')
    image_bytes = pickle.dumps(rgb_image)

    # Get the target index for the specified image
    target_index = lfw_dataset.target[image_index]

    # Check if the target has already reached the limit (5 images)
    if target_index not in target_image_count:
        target_image_count[target_index] = 0

    # Check if the person has less than 10 images and the total number of people is less than 7
    if target_image_count[target_index] < max_images_per_target and len(target_image_count) < max_people:
        # Get the corresponding name from target_names
        name = lfw_dataset.target_names[target_index]

        # Insert the record into the database with target as id
        cursor.execute("INSERT OR IGNORE INTO faces (target, name, image) VALUES (?, ?, ?)", (int(target_index), name, image_bytes))

        # Increment the image count for the target
        target_image_count[target_index] += 1

# Commit the changes to the original database
conn.commit()

# Retrieve data from the augmented database
cursor.execute('SELECT target, name, image FROM faces')
rows = cursor.fetchall()

# Print the total number of images in the augmented dataset
total_images = len(rows)
total_unique_targets = len(set(row[0] for row in rows))

print(f'Total number of images in the dataset: {total_images}')
print(f'Total number of unique targets in the dataset: {total_unique_targets}')

# Close the connection to the database
conn.close()

print(f'Data has been inserted into the "{db_path}" database.')
