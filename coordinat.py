import cv2
import numpy as np
import urllib.request

# Inisialisasi variabel untuk menyimpan koordinat area
areas = []

def get_coordinates(event, x, y, flags, param):
    global areas, current_area

    if event == cv2.EVENT_LBUTTONDOWN:
        # Menambahkan koordinat klik mouse ke daftar area yang sedang aktif
        areas[current_area].append((x, y))
        print(f"Point added to Area {current_area + 1}: ({x}, {y})")

        # Jika sudah mencapai 4 kali klik, tampilkan hasil dan reset daftar area
        if len(areas[current_area]) == 4:
            print(f"Area {current_area + 1} Coordinates:", areas[current_area])
            current_area += 1
            if current_area == 5:
                print("All areas selected.")
                cv2.destroyAllWindows()

# URL untuk MJPEG stream dari ESP32-CAM
url = 'http://192.168.193.1/capture'

cv2.namedWindow('Select Areas')
cv2.setMouseCallback('Select Areas', get_coordinates)

# Inisialisasi daftar area
current_area = 0
areas = [[] for _ in range(5)]

while True:
    # Membaca MJPEG stream dari URL
    img_resp = urllib.request.urlopen(url)
    img_arr = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, -1)

    cv2.imshow("Select Areas", frame)

    # Jika menekan tombol 'esc', keluar dari loop
    if cv2.waitKey(1) & 0xFF == 27 or current_area == 5:
        break

# Menampilkan hasil koordinat area
for i, area in enumerate(areas):
    print(f"area{i + 1} = {area}")

cv2.destroyAllWindows()
