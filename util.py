import cv2
import pandas as pd
import numpy as np
import datetime
import time
from ultralytics import YOLO
import easyocr
import csv
from collections import Counter
import urllib.request
import firebase_admin
from firebase_admin import credentials, firestore
import requests
import os
import glob

cred = credentials.Certificate("E:/Kuliah/Semester 8/Skripsi/Project/skripsi-ba-parkir-99-firebase-adminsdk-ex120-d2bdfcdc2c.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

mitra_uid = "h08P4LepcqgAy0OmVROZxI0ppIw1"

# Initialize the OCR reader
reader = easyocr.Reader(['en'])

# Load model deteksi mobil
model_mobil = YOLO('E:\Kuliah\Semester 8\Skripsi\Project\deteksi_parkir\mobil.pt')

# Load model deteksi plat nomor
model_plat = YOLO('E:\Kuliah\Semester 8\Skripsi\Project\deteksi_parkir\plat.pt')

threshold = 0.3

# Define the license plate conversion function
def convert_license_plate_rules(license_plate_text):
    # Define conversion rules for the license plate parts
    conversion_dict_awal = {
        '0': 'O',
        '1': 'I',
        '2': 'Z',
        '3': 'B',
        '4': 'A',
        '5': 'S',
        # Add more rules as needed
    }

    conversion_dict_tengah = {
        'O': '0',
        'I': '1',
        'Z': '2',
        'B': '3',
        'A': '4',
        'S': '5',
        # Add more rules as needed
    }

    conversion_dict_akhir = {
        '0': 'O',
        '1': 'I',
        '2': 'Z',
        '3': 'B',
        '4': 'A',
        '5': 'S',
        # Add more rules as needed
    }

    # Remove spaces from the license plate text
    license_plate_text = license_plate_text.replace(" ", "")

    # Split the license plate text into parts (awal, tengah, akhir)
    awal, tengah, akhir = license_plate_text[:2], license_plate_text[2:-2], license_plate_text[-2:]

    # Apply conversion rules to each part
    converted_awal = ''.join(conversion_dict_awal.get(char, char) for char in awal)

    # Apply conversion rules to each part
    converted_tengah = ''.join(conversion_dict_tengah.get(char, char) for char in tengah)

    converted_akhir = ''.join(conversion_dict_akhir.get(char, char) for char in akhir)

    # Combine the converted parts
    converted_license_plate = f"{converted_awal} {converted_tengah} {converted_akhir}"

    return converted_license_plate


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

url = 'http://192.168.193.1/capture'

# Set status awal semua jadi off
for area, config in areas.items():
    posisi = config["position"]
    doc_ref = db.collection("mitras").document(mitra_uid).collection("slot").document(posisi)
    doc_ref.update({"status": "off", "dateOn": "", "timeOn": ""})

# Inisialisasi Counter untuk menyimpan jumlah munculnya setiap nomor plat
plat_counter = Counter()
area_image_counter = Counter()

# List untuk menyimpan data CSV
csv_data = []

MAX_IMAGES_TO_CAPTURE = 10  # Set the desired number of images to capture

is_writing_done = False  # Tambahkan variabel penanda
#
# # Initialize a list to store paths of captured license plate images
# plat_crop_image_paths = []

while True:
    img_resp = urllib.request.urlopen(url)
    img_arr = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, -1)

    # Initialize a list to store paths of captured license plate images
    plat_crop_image_paths = []

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

                    results_plat = model_plat(frame)[0]


                    ocr_counter = 0  # Initialize the OCR counter

                    for result_plat in results_plat.boxes.data.tolist():
                        x1_plat, y1_plat, x2_plat, y2_plat, score_plat, class_id_plat = result_plat

                        if score_plat > threshold:
                            cx_plat = int(x1_plat + x2_plat) // 2
                            cy_plat = int(y1_plat + y2_plat) // 2

                            results2_plat = cv2.pointPolygonTest(area_coordinates, ((cx_plat, cy_plat)), False)
                            if results2_plat >= 0:
                                # cv2.rectangle(frame, (int(x1_plat), int(y1_plat)), (int(x2_plat), int(y2_plat)),
                                #               (0, 255, 0), 4)
                                # cv2.putText(frame, results_plat.names[int(class_id_plat)].upper(),
                                #             (int(x1_plat), int(y1_plat - 10)),
                                #             cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

                                plat_crop = frame[int(y1_plat):int(y2_plat), int(x1_plat):int(x2_plat)]
                                # Faktor perbesaran (misalnya, 2x)
                                resize_factor = 3

                                # Menggunakan cv2.resize untuk memperbesar gambar
                                plat_crop_resized = cv2.resize(plat_crop, (0, 0), fx=resize_factor, fy=resize_factor)

                                plat_crop_gray = cv2.cvtColor(plat_crop_resized, cv2.COLOR_BGR2GRAY)

                                _, plat_crop_threshold = cv2.threshold(plat_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                                plat_crop_enhanced = cv2.equalizeHist(plat_crop_gray)
                                plat_crop_smoothed = cv2.GaussianBlur(plat_crop_gray, (5, 5), 0)
                                plat_crop_adaptive_threshold = cv2.adaptiveThreshold(plat_crop_gray, 255,
                                                                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                                                                     cv2.THRESH_BINARY, 11, 2)

                                # Membuat nama file unik dengan nama area dan nomor urut
                                plat_crop_image_path = f'ocr/{area}_{area_image_counter[area] + 1}.png'
                                area_image_counter[area] += 1

                                # Capture and process images only up to the specified limit
                                if area_image_counter[area] <= MAX_IMAGES_TO_CAPTURE:
                                    # Ganti URL dan parameter sesuai dengan konfigurasi ESP32-CAM Anda
                                    lampu_url = 'http://192.168.193.1/control?var=led_intensity&val=200'
                                    requests.get(lampu_url)
                                    # Menyimpan gambar plat nomor dengan nama yang sesuai
                                    cv2.imwrite(plat_crop_image_path, plat_crop_gray)

                                    plat_crop_image_paths.append(plat_crop_image_path)

                                    # Pemanggilan fungsi OCR dengan resolusi tinggi
                                    ocr_results = reader.readtext(
                                        plat_crop_image_path,
                                        detail=0,
                                        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
                                    )

                                    # Jika OCR berhasil mendeteksi plat nomor, simpan hasilnya
                                    if ocr_results:
                                        plat_nomor = ' '.join(ocr_results)  # Mengambil semua hasil OCR

                                        # Konversi plat nomor sebelum menyimpan ke dalam CSV
                                        converted_plate = convert_license_plate_rules(plat_nomor)

                                        # Cek apakah image path sudah ada dalam CSV
                                        existing_entry_index = next(
                                            (i for i, entry in enumerate(csv_data) if entry[1] == plat_crop_image_path),
                                            None)
                                        if existing_entry_index is not None:
                                            # Update plat nomor
                                            csv_data[existing_entry_index][2] = plat_nomor
                                        else:
                                            csv_data.append([area, plat_crop_image_path, converted_plate])

                                        ocr_counter += 1  # Increment the OCR counter
                                        print(plat_nomor)

                                        if area_image_counter[area] == MAX_IMAGES_TO_CAPTURE:
                                            lampu_url = 'http://192.168.193.1/control?var=led_intensity&val=0'
                                            requests.get(lampu_url)

                    # Print the OCR counter after the loop
                    print(f"OCR performed {ocr_counter} times.")

                    # Save the CSV file only if OCR was performed (ocr_counter > 0)
                    if ocr_counter > 0:
                        csv_file_path = 'hasil_ocr.csv'
                        with open(csv_file_path, mode='w', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(['Area', 'Image Path', 'Plat Nomor'])
                            writer.writerows(csv_data)
                        print("CSV file saved.")
                    else:
                        print("No OCR performed. CSV file not saved.")

                    # Load the CSV file into a DataFrame
                    df = pd.DataFrame(csv_data, columns=['Area', 'Image Path', 'Plat Nomor'])

                    # Count the occurrences of each license plate for each area
                    plat_counter = df.groupby(['Area', 'Plat Nomor']).size().reset_index(name='Count')

                    # Print the license plate with the highest frequency for each area
                    for area in df['Area'].unique():
                        area_data = plat_counter[plat_counter['Area'] == area]
                        if not area_data.empty:
                            max_count_row = area_data.loc[area_data['Count'].idxmax()]

                            most_common_plate = max_count_row['Plat Nomor']
                            frequency = max_count_row['Count']
                            total_data = len(df[df['Area'] == area])

                            print(
                                f"Area: {area} - Plat Nomor Paling Banyak Muncul ({total_data} data): {most_common_plate}, Jumlah Muncul: {frequency}")
                            # Simpan data ke Firestore

                            if total_data >= 10:
                                # Update data di Firestore hanya jika sudah ada 10 data
                                plat_doc_name = area
                                mitra_ref = db.collection("mitras").document(mitra_uid)
                                slot_ref = mitra_ref.collection("slot").where("codeSlot", "==", area).stream()



                                # Pastikan hanya ada satu dokumen karena mencocokkan dengan area tertentu
                                for doc in slot_ref:
                                    slot_doc_ref = doc.reference
                                    plat_data = {
                                        "plat": most_common_plate,
                                    }
                                    slot_doc_ref.update(plat_data)
                                    print(f"Data untuk {area} telah diperbarui di Firestore.")
                            else:
                                print(f"Belum cukup data untuk {area}.")

                        else:
                            print(f"Tidak ada plat nomor yang terdeteksi di area {area}.")

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
                if current_time - detection_time > 10:
                    posisi = config["position"]
                    doc_ref = db.collection("mitras").document(mitra_uid).collection("slot").document(posisi)
                    dateOn = datetime.datetime.now().strftime("%Y-%m-%d")
                    timeOn = datetime.datetime.now().strftime("%H:%M:%S")
                    doc_ref.update({"status": "on", "dateOn": dateOn, "timeOn": timeOn})
                    print(f"{area.capitalize()} Terisi")

                    area_filled_status[area] = True

                    # # Reset counter gambar plat nomor jika tidak ada mobil di area
                    # area_image_counter[area] = 0

        else:
            # Reset counter gambar plat nomor jika tidak ada mobil di area
            area_image_counter[area] = 0

            if area_filled_status[area]:
                posisi = config["position"]
                doc_ref = db.collection("mitras").document(mitra_uid).collection("slot").document(posisi)
                doc_ref.update({"status": "off", "dateOn": "", "timeOn": ""})
                print(f"{area.capitalize()} Kosong")
                area_filled_status[area] = False


            # # Hapus data dari Firestore jika area kosong
            # plat_doc_ref = db.collection("mitras").document(mitra_uid).collection("plat").document(area)
            # plat_doc_ref.delete()
            # print(f"Data untuk {area} dihapus dari Firestore karena area kosong.")

            # Hapus semua gambar dalam folder 'ocr' untuk suatu area
            area_image_folder = f'ocr/{area}_*.png'
            area_image_paths = glob.glob(area_image_folder)
            for path in area_image_paths:
                try:
                    os.remove(path)
                    print(f"Image path {path} removed.")
                except Exception as e:
                    print(f"Failed to remove image path {path}: {e}")

            # Hapus isi file CSV yang sesuai dengan area
            csv_data = [entry for entry in csv_data if entry[0] != area]

            # Simpan kembali file CSV tanpa data untuk area yang dihapus
            csv_file_path = 'hasil_ocr.csv'
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Area', 'Image Path', 'Plat Nomor'])
                writer.writerows(csv_data)
            print(f"Data CSV untuk {area} dihapus.")

            mitra_ref = db.collection("mitras").document(mitra_uid)
            slot_ref = mitra_ref.collection("slot").where("codeSlot", "==", area).stream()

            # Pastikan hanya ada satu dokumen karena mencocokkan dengan area tertentu
            for doc in slot_ref:
                slot_doc_ref = doc.reference
                plat_data = {
                    "plat": "",
                }
                slot_doc_ref.update(plat_data)
                print(f"Data untuk {area} telah diperbarui di Firestore.")


    cv2.imshow("Deteksi Parkir", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: #esc
        break

cv2.destroyAllWindows()
