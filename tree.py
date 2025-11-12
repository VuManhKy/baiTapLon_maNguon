import cv2
import numpy as np
import os
import face_recognition

# ===== KHỞI TẠO =====
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

# Thư mục lưu khuôn mặt
os.makedirs("faces", exist_ok=True)

# Nếu có file lưu mã hóa cũ -> tải lại
if os.path.exists("encodings.npy"):
    data = np.load("encodings.npy", allow_pickle=True).item()
    known_encodings = data["encodings"]
    known_names = data["names"]
else:
    known_encodings = []
    known_names = []

# ===== HÀM HỖ TRỢ =====
def register_face(frame, face_location):
    """Cắt và lưu khuôn mặt, đồng thời mã hóa"""
    (top, right, bottom, left) = face_location
    face_img = frame[top:bottom, left:right]
    
    # Nhập tên người dùng
    name = input("Nhập tên cho khuôn mặt này: ").strip()
    if not name:
        print("⚠️ Bỏ qua — không có tên.")
        return

    # Lưu ảnh
    file_path = f"faces/{name}.jpg"
    cv2.imwrite(file_path, face_img)
    print(f"✅ Đã lưu ảnh: {file_path}")

    # Mã hóa khuôn mặt
    encoding = face_recognition.face_encodings(face_img)
    if len(encoding) > 0:
        known_encodings.append(encoding[0])
        known_names.append(name)
        np.save("encodings.npy", {"encodings": known_encodings, "names": known_names})
        print(f"✅ Đã lưu mã hóa cho {name}")
    else:
        print("⚠️ Không tìm thấy khuôn mặt rõ ràng để mã hóa.")

# ===== VÒNG LẶP CHÍNH =====
print("Nhấn 'r' để đăng ký khuôn mặt, 'q' để thoát.")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (face_encoding, (top, right, bottom, left)) in zip(face_encodings, face_locations):
        name = "Unknown"

        # So khớp với khuôn mặt đã lưu
        if known_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_names[best_match_index]

        # Vẽ khung và tên
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    # Đăng ký khuôn mặt
    if key == ord('r'):
        if len(face_locations) > 0:
            register_face(rgb_frame, face_locations[0])
        else:
            print("❌ Không phát hiện khuôn mặt để đăng ký.")
    
    elif key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
