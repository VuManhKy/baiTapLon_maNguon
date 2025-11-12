# dangky.py — Đăng ký khuôn mặt (CPU only) + xử lý SPACE mượt hơn

from pathlib import Path
import json, time, warnings
import numpy as np
import cv2
from insightface.app import FaceAnalysis

# Ẩn cảnh báo rcond từ insightface.transform
warnings.filterwarnings(
    "ignore",
    message="`rcond` parameter will change to the default of machine precision",
    category=FutureWarning
)

# Beep nhỏ khi chụp (chỉ Windows, các HĐH khác tự bỏ qua)
try:
    import winsound
    def beep_ok(): winsound.Beep(1200, 80)
except Exception:
    def beep_ok(): pass

DB = Path("db"); DB.mkdir(exist_ok=True)
EMB = DB / "embeddings.npy"
LAB = DB / "labels.json"

def load_db():
    """Tải embeddings + labels từ đĩa (nếu chưa có thì rỗng)."""
    if EMB.exists() and LAB.exists():
        emb = np.load(EMB)
        labels = json.loads(LAB.read_text(encoding="utf-8"))
    else:
        emb = np.empty((0,512), dtype=np.float32)
        labels = []
    return emb, labels

def save_db(new_emb, new_labels):
    np.save(EMB, new_emb)
    LAB.write_text(json.dumps(new_labels, ensure_ascii=False), encoding="utf-8")

def open_cam():
    """Mở webcam với các backend dự phòng cho Windows."""
    # Thứ tự này thường ổn định nhất trên Win: MSMF -> DSHOW -> ANY
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
    for be in backends:
        cap = cv2.VideoCapture(0, be)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap, be
            cap.release()
    return None, None

def main():
    # --- 1) Khởi tạo InsightFace (CPU only) ---
    try:
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        # ctx_id=0 không ảnh hưởng khi chạy CPU, để nguyên cho thống nhất
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ Đã khởi tạo InsightFace (CPU) thành công.")
    except Exception as e:
        print(f"❌ KHÔNG THỂ KHỞI TẠO INSIGHTFACE: {e}")
        return

    # --- 2) Lấy tên nhân viên trước khi mở cửa sổ (tránh mất focus) ---
    name = input("Nhập tên / mã nhân viên: ").strip()
    if not name:
        print("Tên rỗng → thoát.")
        return

    # --- 3) Mở webcam ---
    cap, be = open_cam()
    if cap is None:
        raise RuntimeError("Không mở được webcam (MSMF/DSHOW/ANY đều thất bại).")
    print(f"📷 Đã mở webcam với backend: {be}")

    win_name = "Enroll"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Warm-up camera ~0.5s
    t0 = time.time()
    while time.time() - t0 < 0.5:
        cap.read()

    print("Nhấn SPACE để chụp mẫu (mục tiêu 15 ảnh). Nhấn ESC để thoát.")
    samples = []
    space_was_down = False   # chống giữ SPACE chụp liên tiếp
    last_status = ""         # tránh spam print

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # đợi một chút rồi thử lại để tránh nuốt CPU
                cv2.waitKey(1)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)

            # Lấy danh sách khuôn mặt
            try:
                faces = app.get(rgb)
            except Exception as e:
                faces = []
                if last_status != "model_error":
                    print("⚠️ Lỗi tạm thời từ InsightFace:", e)
                    last_status = "model_error"

            # Vẽ bbox + tên tạm "Face"
            if faces:
                # lấy mặt lớn nhất (thường là mặt gần)
                f = max(faces, key=lambda fa: (fa.bbox[2]-fa.bbox[0])*(fa.bbox[3]-fa.bbox[1]))
                x1, y1, x2, y2 = map(int, f.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                msg = f"Face detected | {len(samples)}/15"
                cv2.putText(frame, msg, (x1, max(20, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                if last_status != "have_face":
                    print("✅ Đang thấy khuôn mặt. Nhấn SPACE để chụp.")
                    last_status = "have_face"
            else:
                cv2.putText(frame, "No face", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                if last_status != "no_face":
                    print("🔎 Khong thay khuon mat — di chuyen vao khung hinh...")
                    last_status = "no_face"

            cv2.imshow(win_name, frame)

            # Nếu người dùng đóng cửa sổ
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Cửa sổ đóng → thoát.")
                break

            k = cv2.waitKey(1) & 0xFF

            # ESC -> thoát
            if k == 27:
                break

            # SPACE -> chụp khi có mặt; chống giữ phím
            if k == ord(' '):
                if not space_was_down:  # cạnh lên
                    space_was_down = True
                    if faces:
                        emb = faces[0].normed_embedding.astype(np.float32)
                        samples.append(emb)
                        beep_ok()
                        print(f"📸 Đã chụp mẫu: {len(samples)}/15")
                        if len(samples) >= 15:
                            break
                    else:
                        print("⛔ Không có mặt trong khung — không thể chụp.")
            else:
                space_was_down = False

    finally:
        cap.release()
        for _ in range(3):
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    # --- 4) Lưu dữ liệu ---
    if samples:
        emb_db, labels = load_db()
        new_emb = np.vstack([emb_db, np.vstack(samples)]) if emb_db.size else np.vstack(samples)
        new_labels = labels + [name] * len(samples)
        save_db(new_emb, new_labels)
        print(f"✅ Đã lưu {len(samples)} mẫu cho: {name}")
    else:
        print("❗ Chưa lưu mẫu nào.")
        
if __name__ == "__main__":
    main()
