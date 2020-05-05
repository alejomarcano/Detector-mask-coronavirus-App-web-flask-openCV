#!/usr/bin/env python
from flask import Flask, render_template, Response
import io
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import os
import pygame 

pygame.mixer.init()
pygame.mixer.set_num_channels(8)
voice = pygame.mixer.Channel(2)
sonido = pygame.mixer.Sound("audiotapaboca.wav")


app = Flask(__name__)
vc = cv2.VideoCapture(0)



# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.join("face_detector", "deploy.prototxt")
weightsPath = os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model("mask_detector.model")


# def detect_and_predict_mask(frame, faceNet, maskNet):
# 	# grab the dimensions of the frame and then construct a blob
# 	# from it
# 	(h, w) = frame.shape[:2]
# 	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
# 		(104.0, 177.0, 123.0))

# 	# pass the blob through the network and obtain the face detections
# 	faceNet.setInput(blob)
# 	detections = faceNet.forward()

# 	# initialize our list of faces, their corresponding locations,
# 	# and the list of predictions from our face mask network
# 	faces = []
# 	locs = []
# 	preds = []

# 	# loop over the detections
# 	for i in range(0, detections.shape[2]):
# 		# extract the confidence (i.e., probability) associated with
# 		# the detection
# 		confidence = detections[0, 0, i, 2]

# 		# filter out weak detections by ensuring the confidence is
# 		# greater than the minimum confidence
# 		if confidence > 0.5:
# 			# compute the (x, y)-coordinates of the bounding box for
# 			# the object
# 			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
# 			(startX, startY, endX, endY) = box.astype("int")

# 			# ensure the bounding boxes fall within the dimensions of
# 			# the frame
# 			(startX, startY) = (max(0, startX), max(0, startY))
# 			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

# 			# extract the face ROI, convert it from BGR to RGB channel
# 			# ordering, resize it to 224x224, and preprocess it
# 			face = frame[startY:endY, startX:endX]
# 			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
# 			face = cv2.resize(face, (224, 224))
# 			face = img_to_array(face)
# 			face = preprocess_input(face)
# 			face = np.expand_dims(face, axis=0)

# 			# add the face and bounding boxes to their respective
# 			# lists
# 			faces.append(face)
# 			locs.append((startX, startY, endX, endY))

# 	# only make a predictions if at least one face was detected
# 	if len(faces) > 0:
# 		# for faster inference we'll make batch predictions on *all*
# 		# faces at the same time rather than one-by-one predictions
# 		# in the above `for` loop
# 		preds = maskNet.predict(faces)

# 	# return a 2-tuple of the face locations and their corresponding
# 	# locations
# 	return (locs, preds)

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence >0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)
			mask, withoutMask = maskNet.predict(face)[0]
			# add the face and bounding boxes to their respective
			# lists
			#faces.append(face)
			#locs.append((startX, startY, endX, endY))
			label = "Con tapaboca" if mask > withoutMask else "Sin tapaboca"
			color = (0, 255, 0) if label == "Con tapaboca" else (0, 0, 255)
			if label == "Sin tapaboca":
				if voice.get_busy() == False:
					voice.play(sonido)	
		# include the probability in the label
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

			cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen():
	while True:
		read_return_code, frame = vc.read()
		frame = imutils.resize(frame, width=600)
		detect_and_predict_mask(frame, faceNet, maskNet)
		encode_return_code, image_buffer = cv2.imencode('.jpg', frame)
		io_buf = io.BytesIO(image_buffer)
		yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + io_buf.read() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(
        gen(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, threaded=True)
