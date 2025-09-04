import cv2
import numpy as np

# Buka kamera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal membaca frame dari kamera")
        break

    # Ubah ke HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ----------------------------
    # Rentang warna merah (pakai 2 rentang, diperketat)
    lower_red1 = np.array([0, 150, 100])
    upper_red1 = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, 150, 100])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask_red = mask_red1 + mask_red2

    # ----------------------------
    # Rentang warna hijau (diperketat)
    lower_green = np.array([40, 70, 70])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # ----------------------------
    # Rentang warna biru (diperketat)
    lower_blue = np.array([94, 80, 2])
    upper_blue = np.array([126, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # ----------------------------
    # Gabungkan ke dalam list untuk diproses
    masks = [
        ("RED", mask_red, (0, 0, 255)),
        ("GREEN", mask_green, (0, 255, 0)),
        ("BLUE", mask_blue, (255, 0, 0))
    ]

    # Loop setiap warna
    for (label, mask, color) in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000:  # hanya deteksi objek cukup besar
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Tampilkan hasil
    cv2.imshow("Deteksi Warna (Merah, Hijau, Biru)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
