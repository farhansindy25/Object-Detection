# Farhan Sindy
# 151402091
# Sistem Deteksi Objek Manusia


# command untuk menjalankan script: 
# python object-detection.py --graph graph --display 1


# import paket yang dibutuhkan
from mvnc import mvncapi as mvnc
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import datetime, os
import argparse
import numpy as np
import time
import cv2


# membuat direktori baru dengan nama sesuai waktu aplikasi dijalankan
now = datetime.datetime.now()
newDirName = now.strftime("%Y_%m_%d (%H:%M:%S)")
os.mkdir(newDirName)

# inisialisasi list dari class label model

CLASSES = ("background", "aeroplane", "bicycle", "bird",
	"boat", "bottle", "bus", "car", "cat", "chair", "cow",
	"diningtable", "dog", "horse", "motorbike", "manusia",
	"pottedplant", "sheep", "sofa", "train", "tvmonitor")

# inisialisasi list dari daftar class label model yang ingin kita abaikan dalam pendeteksian

IGNORE = set(["background", "aeroplane", "bicycle", "bird",
	"boat","bottle","bus", "car", "cat", "chair", "cow",
	"diningtable", "dog", "horse", "motorbike","pottedplant",
	 "sheep", "sofa", "train", "tvmonitor"])


COLORS = (0, 250, 20)
 
# inisialisasi ukuran frame input dan output
PREPROCESS_DIMS = (300, 300)
DISPLAY_DIMS = (600, 600)

# perhitungan skala ukuran input dan outout
DISP_MULTIPLIER = DISPLAY_DIMS[0] // PREPROCESS_DIMS[0]

def preprocess_image(input_image):
	# preproses gambar
	preprocessed = cv2.resize(input_image, PREPROCESS_DIMS)
	preprocessed = preprocessed - 127.5
	preprocessed = preprocessed * 0.007843
	preprocessed = preprocessed.astype(np.float16)

	return preprocessed

def predict(image, graph):
	
	image = preprocess_image(image)

	# mengirimkan gambar ke NCS untuk mendapatkan prediksi dari jaringan yang ada
	graph.LoadTensor(image, None)
	(output, _) = graph.GetResult()

	# mengambil angka yang tepat dari hasil prediksi objek untuk output 
	# inisialisasi list dengan nama prediksi
	num_valid_boxes = output[0]
	predictions = []

	# loop terhadap hasil
	for box_index in range(num_valid_boxes):
		# calculate the base index into our array so we can extract
		# informasi bounding box
		base_index = 7 + box_index * 7

		# boxes dengan angka non-finite (inf, nan, etc) diabaikan
		if (not np.isfinite(output[base_index]) or
			not np.isfinite(output[base_index + 1]) or
			not np.isfinite(output[base_index + 2]) or
			not np.isfinite(output[base_index + 3]) or
			not np.isfinite(output[base_index + 4]) or
			not np.isfinite(output[base_index + 5]) or
			not np.isfinite(output[base_index + 6])):
			continue

		
		(h, w) = image.shape[:2]
		x1 = max(0, int(output[base_index + 3] * w))
		y1 = max(0, int(output[base_index + 4] * h))
		x2 = min(w,	int(output[base_index + 5] * w))
		y2 = min(h,	int(output[base_index + 6] * h))

		# mengambil prediksi darilabel class, kemungkinan,
		# dan bounding box kordinat (x, y)
		pred_class = int(output[base_index + 1])
		pred_conf = output[base_index + 2]
		pred_boxpts = ((x1, y1), (x2, y2))

		# membuat tuple untuk daftar prediksi
		prediction = (pred_class, pred_conf, pred_boxpts)
		predictions.append(prediction)

	return predictions

# membuat argument parser dan parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-g", "--graph", required=True,
	help="path to input graph file")
ap.add_argument("-c", "--confidence", default=.5,
	help="confidence threshold")
ap.add_argument("-d", "--display", type=int, default=0,
	help="switch to display image on screen")
args = vars(ap.parse_args())

# mengecek movidius
print("[INFO] Mencari perangkat movidius...")
devices = mvnc.EnumerateDevices()

# jika perangkat tidak ditemukan, keluar dari script
if len(devices) == 0:
	print("[INFO] Tidak ada perangkat Movidius NCS yang ditemukan")
	quit()

print("[INFO] Menemukan {} perangkat. perangkat akan digunakan. "
	"membuka perangkat...".format(len(devices)))
device = mvnc.Device(devices[0])
device.OpenDevice()

# membuka file graph CNN
print("[INFO] Memuat graph kedalam memori Raspberry Pi ...")
with open(args["graph"], mode="rb") as f:
	graph_in_memory = f.read()

# memuat graph kedalam NCS
print("[INFO] Mengalokasikan graph ke Movidius NCS...")
graph = device.AllocateGraph(graph_in_memory)

# open a pointer to the video stream thread and allow the buffer to
# start to fill, then start the FPS counter
print("[INFO] Memulai video stream dan penghitungan FPS...")
vs = WebcamVideoStream(0).start()
time.sleep(1)
fps = FPS().start()

count = 0

# loop mengulangi frame dari video yang direkam secara real time
while True:
	try:
		# mengambil frame dari videostream
		# membuat duplikat frame dan melakukan resizing(keperluan tampilan)
		frame = vs.read()
		image_for_result = frame.copy()
		image_for_result = cv2.resize(image_for_result, DISPLAY_DIMS)

		im = image_for_result


		# menggunakan prediksi dari NCS
		predictions = predict(frame, graph)
		b = 0
	
		# loop terhadap prediksi
		for (i, pred) in enumerate(predictions):
			# extrak data prediksi agar dapat dibaca
			(pred_class, pred_conf, pred_boxpts) = pred

			# menyaring prediksi berdasarkan confidence
			if pred_conf > args["confidence"]:
				# print prediksi ke terminal

				print("[INFO] Prediksi #{}: class={}, confidence={}, "
					"boxpoints={}, ".format(i, CLASSES[pred_class], pred_conf,
					pred_boxpts))

				#jika class ada dalam list ignore maka mulai loop dari awal
				idx = int(pred_class)
				if CLASSES[idx] in IGNORE:
					continue

				

				# menampilkan data prediksi ke layar
				if args["display"] > 0:

					b = b+1
					
					# membuat label dari class yang telah terprediksi 
					label = "{}".format(CLASSES[pred_class])

					# extrak informasi dari prediction boxpoints
					(ptA, ptB) = (pred_boxpts[0], pred_boxpts[1])
					ptA = (ptA[0] * DISP_MULTIPLIER, ptA[1] * DISP_MULTIPLIER)
					ptB = (ptB[0] * DISP_MULTIPLIER, ptB[1] * DISP_MULTIPLIER)
					(startX, startY) = (ptA[0], ptA[1])
					y = startY - 15 if startY - 15 > 15 else startY + 15

					# menampilan kotak dan label text
					cv2.rectangle(image_for_result, ptA, ptB,
						COLORS, 2)
					cv2.putText(image_for_result, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS, 3)

					

		#print jumlah objek dari nilai b

		label2 = "Jumlah : {}".format(b)
		font                   = cv2.FONT_HERSHEY_SIMPLEX
		bottomLeftCornerOfText = (15,35)
		fontScale              = 1
		fontColor              = (70, 255,255)
		lineType               = 2

		cv2.putText(image_for_result,label2, 
		bottomLeftCornerOfText, 
		font, 
		fontScale,
		fontColor,
		lineType)

		#jika jumlah objek > 0
		#frame akan disimpan dengan ukuran 300 x 300 
		#kedalam direktori yang baru dibuat saat program dijalankan

		if b>0 :
			im = cv2.resize(image_for_result,(300,300))
			cv2.imwrite("%s/frame%d.jpg" %(newDirName,count), im)     # save frame sebagai file JPEG   
			count += 1

		# menampilkan tampilan pada layar
		if args["display"] > 0:
			# menampilkan frame ke layar
			cv2.imshow("Sistem Deteksi Objek Manusia", image_for_result)
			key = cv2.waitKey(1) & 0xFF

			# jika tombol `q` ditekan, loop akan dihentikan
			if key == ord("q"):
				break

		# update hitungan fps
		fps.update()
	
	# jika "ctrl+c"  ditekan di terminal, hentikan loop
	except KeyboardInterrupt:
		break

	# exeption handling lainnya
	except AttributeError:
		break

# stop mengulang frame
fps.stop()


if args["display"] > 0:
	cv2.destroyAllWindows()

# stop pengambilan video
vs.stop()

# membersihkan graph dan perangkat

# menampilkan informasi tambahan (fps dan waktu rekam)
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
