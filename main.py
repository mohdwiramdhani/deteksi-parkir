import cv2
import pandas as pd
import numpy as np
import datetime
import time
from ultralytics import YOLO
import urllib.request
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("E:/Kuliah/Semester 8/Skripsi/Project/skripsi-ba-parkir-99-firebase-adminsdk-ex120-d2bdfcdc2c.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

mitra_uid = "h08P4LepcqgAy0OmVROZxI0ppIw1"

# Load model deteksi mobil
model_mobil = YOLO('bestV8M45.pt')

my_file = open("E:\Kuliah\Semester 8\Skripsi\Project\deteksi_parkir\labels.txt", "r")
data = my_file.read()
class_list = data.split("\n")

areas_data = db.collection("mitras").document(mitra_uid).collection("slot").stream()

areas = {}

area_filled_status = {}

for doc in areas_data:
    doc_data = doc.to_dict()
    area_name = doc_data["codeSlot"]
    area_position = doc_data["positionSlot"]

    x1y1 = {"dx": doc_data["x1y1"]["dx"], "dy": doc_data["x1y1"]["dy"]}
    x2y2 = {"dx": doc_data["x2y2"]["dx"], "dy": doc_data["x2y2"]["dy"]}
    x3y3 = {"dx": doc_data["x3y3"]["dx"], "dy": doc_data["x3y3"]["dy"]}
    x4y4 = {"dx": doc_data["x4y4"]["dx"], "dy": doc_data["x4y4"]["dy"]}

    area_coordinates = [
        (x1y1["dx"], x1y1["dy"]),
        (x2y2["dx"], x2y2["dy"]),
        (x3y3["dx"], x3y3["dy"]),
        (x4y4["dx"], x4y4["dy"]),
    ]

    areas[area_name] = {"coordinates": area_coordinates, "position": area_position, "detection_time": None}
    area_filled_status[area_name] = False

url = 'http://192.168.155.1/capture'

# Set status awal semua jadi off
for area, config in areas.items():
    posisi = config["position"]
    doc_ref = db.collection("mitras").document(mitra_uid).collection("slot").document(posisi)
    doc_ref.update({"status": "off", "dateOn": "", "timeOn": ""})

while True:
    img_resp = urllib.request.urlopen(url)
    img_arr = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, -1)

    for area, config in areas.items():
        area_coordinates = np.array(config["coordinates"], np.int32)
        is_car_detected = False

        # Deteksi mobil
        results_mobil = model_mobil.predict(frame)
        a_mobil = results_mobil[0].boxes.data
        px_mobil = pd.DataFrame(a_mobil).astype("float")

        for index_mobil, row_mobil in px_mobil.iterrows():
            x1_mobil = int(row_mobil[0])
            y1_mobil = int(row_mobil[1])
            x2_mobil = int(row_mobil[2])
            y2_mobil = int(row_mobil[3])
            d_mobil = int(row_mobil[5])

            c_mobil = class_list[d_mobil]

            if 'mobil' in c_mobil:
                cx_mobil = int(x1_mobil + x2_mobil) // 2
                cy_mobil = int(y1_mobil + y2_mobil) // 2

                results1_mobil = cv2.pointPolygonTest(area_coordinates, ((cx_mobil, cy_mobil)), False)
                if results1_mobil >= 0:
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{area}", (x1_mobil, y1_mobil - 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.circle(frame, (cx_mobil, cy_mobil), 3, (0, 0, 255), -1)
                    cv2.putText(frame, str(c_mobil), (x1_mobil, y1_mobil), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

                    # Set status mobil terdeteksi di dalam area
                    is_car_detected = True

            # Simpan waktu deteksi pertama kali jika belum tersimpan
            if config["detection_time"] is None:
                    config["detection_time"] = time.time()

        cv2.polylines(frame, [area_coordinates], True, (0, 255, 0), 2)

        # Cek apakah mobil ada di dalam area pada iterasi saat ini
        if is_car_detected:
            if not area_filled_status[area]:
                # Cek apakah mobil berada di dalam area selama lebih dari 5 detik
                current_time = time.time()
                detection_time = config["detection_time"]
                if current_time - detection_time > 1:
                    posisi = config["position"]
                    doc_ref = db.collection("mitras").document(mitra_uid).collection("slot").document(posisi)
                    dateOn = datetime.datetime.now().strftime("%Y-%m-%d")
                    timeOn = datetime.datetime.now().strftime("%H:%M:%S")
                    doc_ref.update({"status": "on", "dateOn": dateOn, "timeOn": timeOn})
                    print(f"{area.capitalize()} Terisi")

                    area_filled_status[area] = True

        else:

            if area_filled_status[area]:
                posisi = config["position"]
                doc_ref = db.collection("mitras").document(mitra_uid).collection("slot").document(posisi)
                doc_ref.update({"status": "off", "dateOn": "", "timeOn": ""})
                print(f"{area.capitalize()} Kosong")
                area_filled_status[area] = False


    cv2.imshow("Deteksi Parkir", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: #esc
        break

cv2.destroyAllWindows()
