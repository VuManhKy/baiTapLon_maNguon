#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
delete_face.py — Xóa dữ liệu khuôn mặt đã lưu cho 1 người (hoặc xóa toàn bộ)

Chương trình này làm việc với dữ liệu nhận diện khuôn mặt đã được lưu trong:
  - db/embeddings.npy  : Chứa ma trận vector đặc trưng (embedding) kích thước (N, 512)
  - db/labels.json     : Chứa danh sách tên tương ứng với từng vector trong embeddings.npy
Mỗi lần bạn bấm SPACE lúc đăng ký khuôn mặt → tạo ra 1 embedding và 1 label ứng với tên đó.

Ngoài ra, nếu bạn có lưu ảnh khuôn mặt đã crop thì nằm trong:
  - faces/<Tên>/*.jpg  (Tùy chọn — nếu bạn thêm chức năng lưu ảnh)
"""

from pathlib import Path
import argparse, json, time, shutil
import numpy as np
from collections import Counter

# -----------------------------
# Định nghĩa đường dẫn dữ liệu
# -----------------------------
DB = Path("db")                 # Thư mục chứa dữ liệu nhận diện
EMB = DB / "embeddings.npy"     # File lưu embeddings (vector 512-dim)
LAB = DB / "labels.json"        # File lưu danh sách tên
FACES_DIR = Path("faces")       # (Tùy chọn) thư mục ảnh crop từng người


# -----------------------------
# Hàm tải dữ liệu từ ổ đĩa
# -----------------------------
def load_db():
    # Nếu chưa có thì trả về rỗng
    if not EMB.exists() or not LAB.exists():
        return np.empty((0,512), dtype=np.float32), []
    
    # Đọc embeddings và nhãn từ file
    emb = np.load(EMB)
    labels = json.loads(LAB.read_text(encoding="utf-8"))
    return emb, labels


# -----------------------------
# Hàm ghi dữ liệu cập nhật ra đĩa
# -----------------------------
def save_db(emb, labels):
    DB.mkdir(exist_ok=True)   # Đảm bảo thư mục db tồn tại
    np.save(EMB, emb.astype(np.float32))  # Ghi embeddings
    LAB.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")  # Ghi labels


# -----------------------------
# Tạo bản sao dự phòng trước khi xóa (đề phòng nhầm)
# -----------------------------
def backup_db():
    ts = time.strftime("%Y%m%d-%H%M%S")  # Tạo timestamp
    if EMB.exists():
        shutil.copy2(EMB, EMB.with_name(f"embeddings.backup-{ts}.npy"))
    if LAB.exists():
        shutil.copy2(LAB, LAB.with_name(f"labels.backup-{ts}.json"))


# -----------------------------
# Liệt kê số mẫu theo từng tên trong CSDL
# -----------------------------
def list_counts():
    emb, labels = load_db()
    print(f"Embeddings shape: {emb.shape}, labels: {len(labels)}")
    
    if not labels:
        return
    
    cnt = Counter(labels)  # Đếm số lần từng tên xuất hiện
    for name, c in sorted(cnt.items(), key=lambda x: (-x[1], x[0].lower())):
        print(f"- {name}: {c}")


# -----------------------------
# Xóa toàn bộ mẫu của 1 người theo tên
# -----------------------------
def delete_name(target, delete_images=True, no_backup=False):
    emb, labels = load_db()
    
    if len(labels) == 0:
        print("CSDL trống hoặc chưa khởi tạo.")
        return

    # Tìm tất cả vị trí mẫu có tên trùng target
    idxs = [i for i, lb in enumerate(labels) if lb == target]
    
    if not idxs:
        print(f"Không tìm thấy mẫu nào có nhãn: {target!r}.")
        return

    print(f"Tìm thấy {len(idxs)} mẫu của '{target}'. Đang xóa...")

    # Sao lưu lại CSDL trước khi xóa (trừ khi user tắt auto-backup)
    if not no_backup:
        backup_db()

    # Tạo mask để giữ lại những mẫu không phải của target
    mask = np.ones(len(labels), dtype=bool)
    mask[idxs] = False

    # Áp dụng mask lọc ra dữ liệu còn lại
    kept_emb = emb[mask] if emb.size else np.empty((0,512), dtype=np.float32)
    kept_labels = [labels[i] for i in range(len(labels)) if mask[i]]

    # Lưu lại dữ liệu mới
    save_db(kept_emb, kept_labels)
    print(f"✔ Đã xóa {len(idxs)} mẫu. Còn lại: {kept_emb.shape[0]} mẫu.")

    # Nếu có ảnh crop thì xóa ảnh của người này
    if delete_images:
        person_dir = FACES_DIR / target
        if person_dir.exists():
            shutil.rmtree(person_dir)
            print(f"✔ Đã xóa thư mục ảnh: {person_dir}")
        else:
            print("ⓘ Không có thư mục ảnh của người này (bỏ qua).")


# -----------------------------
# Xóa toàn bộ CSDL (cực kỳ cẩn thận)
# -----------------------------
def delete_all(delete_images=True, no_backup=False):
    emb, labels = load_db()
    
    if not no_backup:
        backup_db()

    # Ghi rỗng hoàn toàn
    save_db(np.empty((0,512), dtype=np.float32), [])
    print("✔ Đã xóa toàn bộ embeddings và labels.")

    # Xóa thư mục ảnh nếu có
    if delete_images and FACES_DIR.exists():
        shutil.rmtree(FACES_DIR)
        print(f"✔ Đã xóa toàn bộ ảnh trong: {FACES_DIR}")


# -----------------------------
# MAIN MENU CLI
# -----------------------------
def main():
    # Tạo câu lệnh CLI thân thiện
    p = argparse.ArgumentParser(description="Xóa dữ liệu khuôn mặt đã lưu.")
    
    # Chọn 1 trong 2: xóa theo tên hoặc xóa toàn bộ
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--name", help="Tên/mã nhân viên cần xóa (xóa tất cả mẫu của tên này).")
    g.add_argument("--all", action="store_true", help="Xóa toàn bộ CSDL (cẩn thận).")
    
    p.add_argument("--list", action="store_true", help="Liệt kê số mẫu theo từng tên trước khi xóa.")
    p.add_argument("--keep-images", action="store_true", help="Không xóa ảnh trong faces/<Tên>.")
    p.add_argument("--no-backup", action="store_true", help="Không tạo file backup trước khi xóa.")
    
    args = p.parse_args()

    if args.list:
        list_counts()
        print("-"*40)

    if args.name:
        delete_name(args.name, delete_images=not args.keep_images, no_backup=args.no_backup)
    elif args.all:
        delete_all(delete_images=not args.keep_images, no_backup=args.no_backup)


if __name__ == "__main__":
    main()
