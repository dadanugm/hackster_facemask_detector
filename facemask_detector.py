import numpy as np
import imutils
import time
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream

face = cv2.dnn.readNet("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
masker = load_model("facemask.model")
vs = VideoStream(src=0).start()
time.sleep(2.0)

def mask_detection(masker, frame, face):
	faces = []
	locs = []
	preds = []
	
	(hight, width) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
	face.setInput(blob)
	detections = face.forward()

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([width,hight,width,hight])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(width - 1, endX), min(hight - 1, endY))

			face_roi = frame[startY:endY, startX:endX]
			face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
			face_roi = cv2.resize(face_roi, (224, 224))
			face_roi = img_to_array(face_roi)
			face_roi = preprocess_input(face_roi)
			face_roi = np.expand_dims(face_roi, axis=0)

			faces.append(face_roi)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		preds = masker.predict(faces)
	return (locs, preds)


def main():
	try:
		while True:
			frame = vs.read()
			frame = imutils.resize(frame, width=400)
			(locs, preds) = mask_detection(masker, frame, face)

			for (box, pred) in zip(locs, preds):
				(startX, startY, endX, endY) = box
				(mask, withoutMask) = pred

				temperature = 35
				if (mask > withoutMask) & (temperature<38):
					label = "T:"+str(temperature)+" ,OK"
					color = (255, 0, 0)
				else:
					label = "T:"+str(temperature)+" ,Failed"
					color = (0, 0, 255)
				cv2.putText(frame, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
				
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF

			if key == ord("q"):
				break
	except:
		time.sleep(0.1)

if __name__ == "__main__":
	while True:
		main()
		cv2.destroyAllWindows()

	

