import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

try:
    cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)  # tạo cửa sổ trước
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.imshow("Webcam", frame)

        k = cv2.waitKey(1) & 0xFF
        # nhấn Q hoặc ESC để thoát
        if k in (ord('q'), 27):
            break

        # nếu người dùng bấm nút X đóng cửa sổ
        if cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE) < 1:
            break
finally:
    # luôn giải phóng dù có lỗi
    cap.release()
    # phá toàn bộ cửa sổ (Windows đôi khi cần gọi vài lần + waitKey)
    for _ in range(3):
        cv2.destroyAllWindows()
        cv2.waitKey(1)
