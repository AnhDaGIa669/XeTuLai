import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# GPIO Pin Configuration
LEFT_MOTOR_FORWARD = 17
LEFT_MOTOR_BACKWARD = 18
RIGHT_MOTOR_FORWARD = 22
RIGHT_MOTOR_BACKWARD = 23

GPIO.setmode(GPIO.BCM)
GPIO.setup(LEFT_MOTOR_FORWARD, GPIO.OUT)
GPIO.setup(LEFT_MOTOR_BACKWARD, GPIO.OUT)
GPIO.setup(RIGHT_MOTOR_FORWARD, GPIO.OUT)
GPIO.setup(RIGHT_MOTOR_BACKWARD, GPIO.OUT)

def stop():
    """ Stop the car """
    GPIO.output(LEFT_MOTOR_FORWARD, GPIO.LOW)
    GPIO.output(LEFT_MOTOR_BACKWARD, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_FORWARD, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_BACKWARD, GPIO.LOW)

def go_straight():
    """ Move forward """
    GPIO.output(LEFT_MOTOR_FORWARD, GPIO.HIGH)
    GPIO.output(LEFT_MOTOR_BACKWARD, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_FORWARD, GPIO.HIGH)
    GPIO.output(RIGHT_MOTOR_BACKWARD, GPIO.LOW)

def turn_left():
    """ Turn left """
    GPIO.output(LEFT_MOTOR_FORWARD, GPIO.LOW)
    GPIO.output(LEFT_MOTOR_BACKWARD, GPIO.HIGH)
    GPIO.output(RIGHT_MOTOR_FORWARD, GPIO.HIGH)
    GPIO.output(RIGHT_MOTOR_BACKWARD, GPIO.LOW)

def turn_right():
    """ Turn right """
    GPIO.output(LEFT_MOTOR_FORWARD, GPIO.HIGH)
    GPIO.output(LEFT_MOTOR_BACKWARD, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_FORWARD, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_BACKWARD, GPIO.HIGH)

def execute_command(command):
    """ Execute movement command """
    print(f"Executing Command: {command}")
    if command == "Go Straight":
        go_straight()
    elif command == "Turn Left":
        turn_left()
    elif command == "Turn Right":
        turn_right()
    else:
        stop()
    time.sleep(1)
    stop()

def color_threshold(image):
    """ Apply color thresholding to detect white lane markings """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([179, 60, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    return mask_white

def perspective_transform(image):
    """ Apply perspective transformation """
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
    warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
    return warped

def detect_lane(frame):
    """ Detect lane lines in the frame """
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
    
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)  
            if slope < -0.3:
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.3:
                right_lines.append((x1, y1, x2, y2))
    
    return left_lines, right_lines, warped, edges, edges_closed

def control_vehicle(left_lines, right_lines):
    """ Determine movement direction and control the vehicle """
    if len(left_lines) > 0 and len(right_lines) > 0:
        command = "Go Straight"
    elif len(left_lines) > 0:
        command = "Turn Right"
    elif len(right_lines) > 0:
        command = "Turn Left"
    else:
        command = "Stop"
    
    print(f"Executing Command: {command}")
    execute_command(command)
    
    return command

############################
# Main Program
############################
cap = cv2.VideoCapture(0)  # Use the camera

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        left_lines, right_lines, warped, edges, edges_closed = detect_lane(frame)
        direction = control_vehicle(left_lines, right_lines)

        # Draw lane detection results
        lane_warped = warped.copy()
        for line in left_lines + right_lines:
            x1, y1, x2, y2 = line
            cv2.line(lane_warped, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Display movement direction
        cv2.putText(frame, direction, (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show processed frames
        small_width = frame.shape[1] // 3
        small_height = frame.shape[0] // 3

        frame_resized = cv2.resize(frame, (small_width, small_height))
        warped_resized = cv2.resize(warped, (small_width, small_height))
        edges_resized = cv2.resize(edges, (small_width, small_height))
        closed_resized = cv2.resize(edges_closed, (small_width, small_height))
        lane_resized = cv2.resize(lane_warped, (small_width, small_height))

        edges_resized = cv2.cvtColor(edges_resized, cv2.COLOR_GRAY2BGR)
        closed_resized = cv2.cvtColor(closed_resized, cv2.COLOR_GRAY2BGR)

        top_row = np.hstack((frame_resized, warped_resized, edges_resized))
        bottom_row = np.hstack((closed_resized, lane_resized, np.zeros_like(lane_resized)))
        combined_display = np.vstack((top_row, bottom_row))

        cv2.imshow("Lane Detection & Control", combined_display)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopping program!")
finally:
    stop()
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
