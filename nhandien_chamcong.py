# nhan_dien_chamcong.py
# Chương trình nhận diện khuôn mặt và ghi chấm công tự động

from pathlib import Path         # Quản lý đường dẫn file/thư mục tiện hơn
import json, time, csv           # json dùng đọc tên, time lấy thời gian, csv ghi file csv
import numpy as np               # Xử lý mảng & vector embeddings
import cv2                       # OpenCV: xử lý ảnh + webcam
from insightface.app import FaceAnalysis  # Model nhận diện khuôn mặt

# ----------------- ĐƯỜNG DẪN CƠ SỞ DỮ LIỆU -----------------
DB = Path("db")                              # thư mục chứa embeddings + labels
ATT = Path("attendance"); ATT.mkdir(exist_ok=True)   # thư mục lưu bảng chấm công
EMB = DB / "embeddings.npy"                 # file vector embeddings
LAB = DB / "labels.json"                    # file tên tương ứng embeddings

THRESH = 0.40  # Ngưỡng quyết định nhận ra hay Unknown (thấp hơn = dễ nhận hơn)

# ----------------- HÀM TẢI CƠ SỞ DỮ LIỆU -----------------
def load_db():
    # Kiểm tra nếu thiếu dữ liệu → báo lỗi
    if not (EMB.exists() and LAB.exists()):
        raise RuntimeError("Chưa có dữ liệu. Hãy chạy dangky.py trước.")
    X = np.load(EMB)  # (N,512) mảng embeddings
    y = json.loads(LAB.read_text(encoding="utf-8"))  # danh sách tên
    return X, y

# ----------------- TÍNH COSINE SIMILARITY -----------------
def cosine_sim(a, B):  # a:(512,)  B:(N,512)
    # Nhân ma trận → trả về mức giống nhau giữa khuôn mặt hiện tại & CSDL
    return (B @ a)

# ----------------- GHI DỮ LIỆU CHẤM CÔNG -----------------
def mark_attendance(person):
    date = time.strftime("%Y%m%d")           # tên file dạng YYYYMMDD
    fpath = ATT / f"att_{date}.csv"          # ví dụ: attendance/att_20251111.csv

    weekday = time.strftime("%A")            # Thứ (English)
    date_str = time.strftime("%Y-%m-%d")     # Ngày đầy đủ
    time_str = time.strftime("%H:%M:%S")     # Giờ hiện tại
    
    first = not fpath.exists()               # Nếu file chưa từng được tạo
    with fpath.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        
        if first:                            # Tạo dòng tiêu đề lần đầu
            w.writerow(["weekday", "date", "time", "name"])
        
        # Ghi 1 dòng chấm công
        w.writerow([weekday, date_str, time_str, person])

    print(f"✅ Đã chấm công: {person} lúc {time_str} ngày {date_str} ({weekday})")

# ----------------- CHƯƠNG TRÌNH CHÍNH -----------------
def main():
    X, y = load_db()   # Tải embeddings + tên tương ứng

    # Khởi tạo mô hình InsightFace, ưu tiên chạy GPU → nhanh hơn
    app = FaceAnalysis(
        name='buffalo_l',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(640, 640))   # ctx_id=0 → GPU chính

    # Mở webcam (DShow tối ưu cho Windows)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # đặt độ phân giải
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("Không mở được webcam")

    last_time = {}   # dùng để không chấm công liên tục
    COOLDOWN = 120    # mỗi người cách nhau 120 giây mới chấm lại

    while True:
        ok, frame = cap.read()   # đọc 1 frame từ cam
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # đổi màu cho model
        rgb = np.ascontiguousarray(rgb)               # tối ưu bộ nhớ
        faces = app.get(rgb)                          # detect + extract embedding

        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)         # bounding box khuôn mặt
            emb = f.normed_embedding.astype(np.float32)  # embedding 512-dim

            sims = cosine_sim(emb, X)                 # tính độ giống với CSDL
            idx = int(np.argmax(sims))                # lấy mẫu giống nhất
            score = float(sims[idx])                  # mức giống
            name = y[idx] if score >= THRESH else "Unknown"  # quyết định tên

            # Vẽ khung và tên lên ảnh
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} {score:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Nếu đã nhận ra và đủ thời gian cooldown → ghi công
            if name != "Unknown":
                now = time.time()
                if name not in last_time or now - last_time[name] > COOLDOWN:
                    mark_attendance(name)
                    last_time[name] = now

        cv2.imshow("Attendance", frame)               # hiển thị camera
        if (cv2.waitKey(1) & 0xFF) == 27:             # ESC → thoát
            break

    cap.release()                                     # giải phóng cam
    cv2.destroyAllWindows()

# ----------------- CHẠY CHƯƠNG TRÌNH -----------------
if __name__ == "__main__":
    main()
