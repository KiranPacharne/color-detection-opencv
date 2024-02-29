import cv2
import numpy as np

# Define the lower and upper bounds of each color in HSV format
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])

lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Calibration information
pixels_per_cm = 10  # Assuming 10 pixels per cm

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        break

    # Get the dimensions of the frame
    height, width, _ = frame.shape

    # Get the center of the frame
    center_x = width // 2
    center_y = height // 2

    # Calculate the diameter of the circular region in pixels
    diameter_cm = 3
    diameter_pixels = int(diameter_cm * pixels_per_cm)

    # Define the circular region around the crosshair
    cv2.circle(frame, (center_x, center_y), diameter_pixels // 2, (0, 0, 0), 2)

    # Extract the circular region around the crosshair
    region_x1 = max(0, center_x - diameter_pixels // 2)
    region_y1 = max(0, center_y - diameter_pixels // 2)
    region_x2 = min(width, center_x + diameter_pixels // 2)
    region_y2 = min(height, center_y + diameter_pixels // 2)
    region = frame[region_y1:region_y2, region_x1:region_x2]

    # Convert the region from BGR to HSV color space
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # Threshold the HSV region to get only the blue color
    blue_mask = cv2.inRange(hsv_region, lower_blue, upper_blue)

    # Threshold the HSV region to get only the red color
    red_mask1 = cv2.inRange(hsv_region, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_region, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Threshold the HSV region to get only the green color
    green_mask = cv2.inRange(hsv_region, lower_green, upper_green)

    # Combine all color detections
    color_mask = cv2.bitwise_or(blue_mask, cv2.bitwise_or(red_mask, green_mask))

    # Create an output frame of the same size as the original frame
    output_frame = np.zeros_like(frame)

    # Place the color-detected region at the appropriate location within the output frame
    output_frame[region_y1:region_y2, region_x1:region_x2] = cv2.bitwise_and(region, region, mask=color_mask)

    color_names = []
    if np.any(blue_mask):
        color_names.append('Blue')
    if np.any(red_mask):
        color_names.append('Red')
    if np.any(green_mask):
        color_names.append('Green')

    # Display the names of the detected colors on the output frame
    text = ', '.join(color_names) if color_names else 'No color detected'
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Display the original frame and result
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Color Detected', output_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('!'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
