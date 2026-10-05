import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    print(ret, frame)

    if not ret:
        break

    cv2.imshow('Live Webcam', frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()