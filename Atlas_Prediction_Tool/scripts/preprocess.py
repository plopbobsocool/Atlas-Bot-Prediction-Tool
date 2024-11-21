import cv2
import os
import numpy as np
import pandas as pd

# Input and output paths
input_folder = 'data/maps'
output_file = 'data/features.csv'

features_list = []

# Loop through the images
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):  # Assuming images are in PNG format
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        
        # Calculate the average color for the image
        avg_red = np.mean(image[:, :, 0])  # For RGB image, channel 0 is red
        avg_green = np.mean(image[:, :, 1])  # Channel 1 is green
        avg_blue = np.mean(image[:, :, 2])  # Channel 2 is blue
        
        # Create the grid_id (use the filename and coordinates as an example)
        grid_id = filename.split('.')[0]  # Assuming the filename is the grid_id
        
        # Append grid_id and features
        features_list.append([grid_id, avg_red, avg_green, avg_blue])

# Convert the list to a DataFrame and add column headers
columns = ['grid_id', 'avg_red', 'avg_green', 'avg_blue']
df = pd.DataFrame(features_list, columns=columns)

# Save the DataFrame to CSV
df.to_csv(output_file, index=False)

print(f"Features saved to {output_file}")
