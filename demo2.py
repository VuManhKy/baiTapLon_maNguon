import cv2
import os
import numpy as np
import csv
from datetime import datetime

# ====== CẤU HÌNH ======
DATASET_DIR = "faces_dataset"
MODEL_FILE = "lbph_model.yml"
ATTENDANCE_FILE = "attendance.csv"
CAMERA_INDEX = 0
IMG_SIZE = (160, 160)
NUM_IMAGES = 30
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# ====================== GHI KHUÔN MẶT ======================
def capture_faces(person_name):
    os.makedirs(os.path.join(DATASET_DIR, person_name), exist_ok=True)
    cam = cv2.VideoCapture(CAMERA_INDEX)
    detector = cv2.CascadeClassifier(CASCADE_PATH)

    count = 0
    print(f"[INFO] Ghi khuôn mặt cho '{person_name}'... Nhấn Q để dừng.")
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, IMG_SIZE)
            cv2.imwrite(f"{DATASET_DIR}/{person_name}/{person_name}_{count}.jpg", face_img)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {count}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Ghi khuon mat", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= NUM_IMAGES:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Đã lưu {count} ảnh cho {person_name}.")


# ====================== TRAIN MODEL ======================
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_dict = {}
    label_id = 0

    print("[INFO] Đang train model...")

    for person_name in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_path):
            continue
        label_dict[label_id] = person_name

        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(label_id)

        label_id += 1

    if not faces:
        print("[ERROR] Không có dữ liệu khuôn mặt để train.")
        return

    recognizer.train(faces, np.array(labels))
    recognizer.write(MODEL_FILE)

    # Lưu label mapping
    with open("labels.txt", "w", encoding="utf-8") as f:
        for k, v in label_dict.items():
            f.write(f"{k}:{v}\n")

    print("[INFO] Train hoàn tất. Model lưu tại", MODEL_FILE)


# ====================== LOAD LABELS ======================
def load_labels(file_path="labels.txt"):
    label_dict = {}
    if not os.path.exists(file_path):
        return label_dict
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            k, v = line.strip().split(":")
            label_dict[int(k)] = v
    return label_dict


# ====================== GHI ĐIỂM DANH ======================
def record_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Tạo file nếu chưa có
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

    # Kiểm tra xem đã điểm danh hôm nay chưa
    with open(ATTENDANCE_FILE, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
        for row in rows:
            if len(row) >= 2 and row[0] == name and row[1] == date_str:
                return  # đã ghi hôm nay rồi

    # Ghi thêm dòng mới
    with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([name, date_str, time_str])
    print(f"[ATTENDANCE] {name} đã điểm danh lúc {time_str} ({date_str})")


# ====================== NHẬN DIỆN KHUÔN MẶT ======================
def recognize_faces():
    if not os.path.exists(MODEL_FILE):
        print("[ERROR] Chưa có model. Hãy train trước!")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)
    labels = load_labels()

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cam = cv2.VideoCapture(CAMERA_INDEX)
    print("[INFO] Bắt đầu nhận diện. Nhấn Q để thoát.")

    recognized_today = set()

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, IMG_SIZE)
            label_id, confidence = recognizer.predict(face_img)
            name = labels.get(label_id, "Unknown")

            if confidence < 80:
                color = (0, 255, 0)
                text = f"{name} ({confidence:.0f})"
                if name not in recognized_today:
                    record_attendance(name)
                    recognized_today.add(name)
            else:
                color = (0, 0, 255)
                text = "Unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Nhan dien khuon mat", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


# ====================== MENU ======================
def main():
    while True:
        print("\n====== MENU ======")
        print("1. Ghi khuôn mặt mới")
        print("2. Train model")
        print("3. Nhận diện & điểm danh")
        print("0. Thoát")
        choice = input("Chọn: ")

        if choice == "1":
            name = input("Nhập tên người: ").strip()
            if name:
                capture_faces(name)
        elif choice == "2":
            train_model()
        elif choice == "3":
            recognize_faces()
        elif choice == "0":
            break
        else:
            print("Lựa chọn không hợp lệ.")


if __name__ == "__main__":
    main()
