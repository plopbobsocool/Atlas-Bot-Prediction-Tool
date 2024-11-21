import cv2
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("models/rust_build_predictor.pkl")

# Load a new map image to predict build spots
new_map_image = cv2.imread("data/maps/2024-11-10.jpg")

# Define grid size and dimensions
grid_size = 30
height, width, _ = new_map_image.shape
cell_height = height // grid_size
cell_width = width // grid_size

# Placeholder for predictions
predictions = []

# Process each grid cell
for row in range(grid_size):
    for col in range(grid_size):
        cell = new_map_image[
            row * cell_height : (row + 1) * cell_height,
            col * cell_width : (col + 1) * cell_width,
        ]
        
        # Extract features for the cell (same as training features)
        avg_color = np.mean(cell, axis=(0, 1))
        features = np.array(avg_color).reshape(1, -1)
        
        # Predict if this cell is likely a build spot
        prediction = model.predict(features)[0]
        
        # Store the grid cell prediction
        predictions.append((chr(65 + col) + str(row + 1), prediction))

# Convert predictions to DataFrame
predictions_df = pd.DataFrame(predictions, columns=["grid_id", "prediction"])

# Visualize the map with predictions
plt.imshow(new_map_image)
for grid_id, prediction in predictions_df.values:
    if prediction == 1:
        # Mark build spots in red
        col, row = ord(grid_id[0]) - 65, int(grid_id[1]) - 1
        plt.plot(col * cell_width + cell_width // 2, row * cell_height + cell_height // 2, "ro")

plt.show()

# Save predictions to CSV
predictions_df.to_csv("output/predictions.csv", index=False)
print("Predictions complete and visualized!")
