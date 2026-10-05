import cv2
import numpy as np

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     print(ret, frame)

#     if not ret:
#         break

#     cv2.imshow('Live Webcam', frame)
#     if cv2.waitKey(40) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# ==============================================================================
# 1. Real-time color detection using webcam for yellow
# ==============================================================================

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    hsv_video = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Lower and upper bounds for yellow color in HSV
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    # Create a mask to isolate yellow color
    mask = cv2.inRange(hsv_video, yellow_lower, yellow_upper)
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(f"Area: {area}")

        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Area: {int(area)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imshow('Mask Biner', mask)
    cv2.imshow('Yellow Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()