import cv2
import os

# Input and output paths
input_folder = "data/maps"
output_folder = "data/processed_maps"
grid_size = 30  # 30x30 grid

os.makedirs(output_folder, exist_ok=True)

# Process each map
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the image
        map_path = os.path.join(input_folder, filename)
        map_image = cv2.imread(map_path)
        
        # Get dimensions and calculate grid cell size
        height, width, _ = map_image.shape
        cell_height = height // grid_size
        cell_width = width // grid_size

        # Divide into grid cells
        for row in range(grid_size):
            for col in range(grid_size):
                cell = map_image[
                    row * cell_height : (row + 1) * cell_height,
                    col * cell_width : (col + 1) * cell_width,
                ]
                # Save each grid cell with its ID
                grid_id = chr(65 + col) + str(row + 1)  # A1, B1, etc.
                output_path = os.path.join(output_folder, f"{filename}_{grid_id}.png")
                cv2.imwrite(output_path, cell)

print("Preprocessing complete!")
