import json
import numpy as np

# Load the JSON
with open("Backend/Cattle_Labelme/Amritmahal_5.json", "r") as f:
    data = json.load(f)

for shape in data['shapes']:
    points = np.array(shape['points'])  # Nx2 array of x,y points

    # Bounding box
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    width  = x_max - x_min
    height = y_max - y_min

    # Centroid (center of polygon)
    centroid_x = points[:,0].mean()
    centroid_y = points[:,1].mean()

    # Polygon area (Shoelace formula)
    x = points[:,0]
    y = points[:,1]
    area = 0.5 * np.abs(np.dot(x,np.roll(y,1)) - np.dot(y,np.roll(x,1)))

    # Aspect ratio
    aspect_ratio = width / height

    print(f"Label: {shape['label']}")
    print(f"Bounding Box: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
    print(f"Width: {width:.2f}, Height: {height:.2f}")
    print(f"Centroid: ({centroid_x:.2f}, {centroid_y:.2f})")
    print(f"Polygon Area: {area:.2f} pixels^2")
    print(f"Aspect Ratio: {aspect_ratio:.2f}")
