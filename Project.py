import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands and Drawing Utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture and set up canvas
cap = cv2.VideoCapture(0)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Variables for pen color, thickness, and eraser
pen_color = (255, 0, 0)  # Default: blue
pen_thickness = 5        # Default thickness
eraser_thickness = 70
small_eraser_thickness = 15

# Variable to store the previous drawing point for smooth lines
previous_point = None  

# Define colors
colors = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'black': (0, 0, 0)
}

# Initialize MediaPipe Hands
with mp_hands.Hands(max_num_hands=3, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and get hand landmarks
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get landmark positions
                landmarks = hand_landmarks.landmark
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                middle_tip = landmarks[12]
                ring_tip = landmarks[16]
                pinky_mcp = landmarks[17]

                # Convert landmarks to pixel positions
                thumb_tip_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_tip_pos = (int(index_tip.x * w), int(index_tip.y * h))
                middle_tip_pos = (int(middle_tip.x * w), int(middle_tip.y * h))
                ring_tip_pos = (int(ring_tip.x * w), int(ring_tip.y * h))
                pinky_mcp_pos = (int(pinky_mcp.x * w), int(pinky_mcp.y * h))

                # Drawing Mode (continuous line): Thumb + Index/Middle/Ring Tip
                if (np.linalg.norm(np.array(thumb_tip_pos) - np.array(index_tip_pos)) < 30 or
                    np.linalg.norm(np.array(thumb_tip_pos) - np.array(middle_tip_pos)) < 30 or
                    np.linalg.norm(np.array(thumb_tip_pos) - np.array(ring_tip_pos)) < 30):
                    
                    if previous_point:  # Draw from the previous point to current
                        cv2.line(canvas, previous_point, thumb_tip_pos, pen_color, pen_thickness)
                    previous_point = thumb_tip_pos  # Update previous point

                # Eraser Mode (Large): Thumb + Pinky MCP
                elif np.linalg.norm(np.array(thumb_tip_pos) - np.array(pinky_mcp_pos)) < 30:
                    cv2.circle(frame, thumb_tip_pos, eraser_thickness, (0, 0, 0), -1)
                    cv2.circle(canvas, thumb_tip_pos, eraser_thickness, (0, 0, 0), -1)
                    previous_point = None  # Reset previous point

                # Change Pen Color: Based on Thumb + Tip Gestures
                if np.linalg.norm(np.array(thumb_tip_pos) - np.array(index_tip_pos)) < 30:
                    pen_color = colors['red']  # Red
                elif np.linalg.norm(np.array(thumb_tip_pos) - np.array(middle_tip_pos)) < 30:
                    pen_color = colors['green']  # Green
                elif np.linalg.norm(np.array(thumb_tip_pos) - np.array(ring_tip_pos)) < 30:
                    pen_color = colors['blue']  # Blue

                # Reset previous point to avoid dotted lines when not drawing
                else:
                    previous_point = None

        # Overlay the canvas on the frame
        frame1 = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
        
        # Display the frame
        cv2.imshow('Virtual Whiteboard', frame1)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Clear the canvas
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        elif key == ord('q'):  # Exit the program
            break

cap.release()
cv2.destroyAllWindows()