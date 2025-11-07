import cv2
import numpy as np

# Load bộ nhận diện khuôn mặt có sẵn trong OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ảnh overlay (ví dụ: kính, mũ, sticker, v.v.)
overlay_img = cv2.imread('image.png', cv2.IMREAD_UNCHANGED)  # đọc cả kênh alpha

# Mở webcam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        overlay_y = y - int(h * 0.8)  # dịch lên trên 30% chiều cao
        overlay_y = max(0, overlay_y)  # không cho vượt khỏi khung ảnh

        # Resize ảnh overlay để khớp với khuôn mặt
        overlay_resized = cv2.resize(overlay_img, (w, h))

        # Nếu ảnh overlay có kênh alpha (trong suốt)
        if overlay_resized.shape[2] == 4:
            alpha_s = overlay_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(3):
                frame[overlay_y:overlay_y+h, x:x+w, c] = (alpha_s * overlay_resized[:, :, c] +
                                          alpha_l * frame[overlay_y:overlay_y+h, x:x+w, c])
        else:
            # Ảnh không có alpha thì chèn thẳng (đè hoàn toàn)
            frame[y:y+h, x:x+w] = overlay_resized

    cv2.imshow('lmao', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
