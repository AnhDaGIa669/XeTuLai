import socket
import cv2
import numpy as np
import time

# Raspberry Pi IP address (Update with the actual IP of the Pi)
RASPBERRY_IP = "100.112.26.19"
PORT = 12345

# Variables to count curve detections
left_curve_count = 0
right_curve_count = 0
curve_threshold = 5

def send_command(command):
    """Send control command to Raspberry Pi via socket"""
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((RASPBERRY_IP, PORT))
        client.sendall(command.encode())
        client.close()
        print(f"Sent command: {command}")  # Print to terminal for debugging
    except Exception as e:
        print(f"Error sending command: {e}")

def color_threshold(image):
    """Filter white color in HSV image"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([179, 60, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower_white, upper_white)

def perspective_transform(image):
    """Apply perspective transformation"""
    height, width = image.shape[:2]
    src_points = np.float32([
        [int(0.1 * width), int(0.55 * height)],
        [int(0.9 * width), int(0.55 * height)],
        [0, height],
        [width, height]
    ])
    dst_points = np.float32([
        [0, 0],
        [width, 0],
        [0, height],
        [width, height]
    ])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)

def detect_lane(frame):
    """Process image to detect lane lines"""
    mask_color = color_threshold(frame)
    color_filtered = cv2.bitwise_and(frame, frame, mask=mask_color)
    warped = perspective_transform(color_filtered)

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    lines = cv2.HoughLinesP(edges_closed, 1, np.pi / 180, threshold=50,
                            minLineLength=50, maxLineGap=50)
    
    left_lines, right_lines = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero
            if slope < -0.3:
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.3:
                right_lines.append((x1, y1, x2, y2))
    
    return left_lines, right_lines, warped, edges, edges_closed, mask_color, color_filtered

def control_vehicle(left_lines, right_lines):
    """Decide vehicle movement direction"""
    global left_curve_count, right_curve_count
    
    if len(left_lines) > 0 and len(right_lines) > 0:
        left_curve_count = 0
        right_curve_count = 0
        return "Go Straight"
    elif len(left_lines) > 0:
        left_curve_count += 1
        right_curve_count = 0
        if left_curve_count >= curve_threshold:
            left_curve_count = 0
            return "Turn Right"
    elif len(right_lines) > 0:
        right_curve_count += 1
        left_curve_count = 0
        if right_curve_count >= curve_threshold:
            right_curve_count = 0
            return "Turn Left"
    else:
        left_curve_count = 0
        right_curve_count = 0
        return "Stop"

# Main program
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    left_lines, right_lines, warped, edges, edges_closed, mask_color, color_filtered = detect_lane(frame)
    direction = control_vehicle(left_lines, right_lines)
    print(f"Movement direction: {direction}")

    # Automatically send command to Raspberry Pi
    if direction != "Go Straight":
        send_command(direction)
    
    # Display processed images
    lane_warped = warped.copy()
    for line in left_lines + right_lines:
        x1, y1, x2, y2 = line
        cv2.line(lane_warped, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.putText(lane_warped, direction, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Combine images for display
    mask_color_bgr = cv2.cvtColor(mask_color, cv2.COLOR_GRAY2BGR)  # Convert mask image to 3 channels
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_closed_bgr = cv2.cvtColor(edges_closed, cv2.COLOR_GRAY2BGR)

    top_row = np.hstack((frame, color_filtered, mask_color_bgr))
    bottom_row = np.hstack((warped, edges_bgr, edges_closed_bgr))
    combined_view = np.vstack((top_row, bottom_row))

    # Resize for better display
    combined_view = cv2.resize(combined_view, (900, 600))

    cv2.imshow("Lane Detection - Multi View", combined_view)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
