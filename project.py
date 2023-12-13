import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('kemeja.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use Canny edge detector to find edges in the image
edges = cv2.Canny(gray, 50, 150)

# Find contours in the edged image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through the contours and filter based on area
for contour in contours:
    area = cv2.contourArea(contour)

    # You may need to adjust this threshold based on your specific case
    if area > 100:
        # Draw a bounding box around the detected object
        x, y, w, h = cv2.boundingRect(contour)

        # Check if the aspect ratio is close to 1 (indicating a line)
        aspect_ratio = w / h
        if 0.2 < aspect_ratio < 2.2:
             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
# Create a 1x3 subplot grid
plt.figure(figsize=(12, 4))

# Plot the original image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Plot the grayscale image
plt.subplot(1, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')

# Plot the edges
plt.subplot(1, 3, 3)
plt.imshow(edges, cmap='gray')
plt.title('Edges')

# Display the result
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
