#Import Libraries. OpenCV for processing and deep-learning tasks
import cv2
#Standard Math operations
import math
#For parsing arguments in the command prompt.
import argparse

#Age detection function.
def highlightFace(net, frame, conf_threshold=0.7):
	frameOpencvDnn = frame.copy()
	frameHeight = frameOpencvDnn.shape[0]
	frameWidth = frameOpencvDnn.shape[1]
	#Blob creation. Makes it suitable to be read by the model.
	#Resize it to 300*300 pixels, subtract mean values [104, 117, 123] for preprocessing, return [1, 3, 300, 300]
	blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
	
	#Detection, model processes the blob and gives an output.
	net.setInput(blob)
	detections = net.forward()
	faceBoxes = []

	#Consider thresholds that have a confidence level of >0.7 to reduce false positives
	for i in range (detections.shape[2]):
		confidence = detections[0,0,i,2]
		if confidence>conf_threshold:
			#Return Values.
			x1 = int(detections[0,0,i,3]*frameWidth)
			y1 = int(detections[0,0,i,4]*frameHeight)
			x2 = int(detections[0,0,i,5]*frameWidth)
			y2 = int(detections[0,0,i,6]*frameHeight)
			faceBoxes.append([x1,y1,x2,y2])
			cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), [0,255,0], int(round(frameHeight/150)), 8)
	return frameOpencvDnn, faceBoxes

#Initialize Argparser
parser = argparse.ArgumentParser()
#Accept --image argument to specify input image
parser.add_argument("--image")

args = parser.parse_args()

#Face detection
faceProto = "opencv_face_detector.pbtxt"
faceModel ="opencv_face_detector_uint8.pb"

#Age Detection
ageProto = "age_model_deploy.prototxt"
ageModel = "age_net.caffemodel"

#Gender Detection
genderProto = "gender_model_deploy.prototxt"
genderModel = "gender_net.caffemodel"

#Mean Values
Model_Mean_Values = (78.4263377603, 87.7689143744, 114.895847746)

#Age list
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

#Gender List
genderList = ['Male', 'Female']

#Read face, age, gender
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

#Video Processing Loop where the script processes image in a loop until the user presses a key.
video = cv2.VideoCapture(args.image if args.image else 0)
#Add buffer around the image when Region of Interest is detected.
padding = 20

#Frame reading if it's a video.
while cv2.waitKey(1)<0:
	hasFrame, frame = video.read()
	#Framer reading if it's not a video.
	if not hasFrame:
		cv2.waitKey()
		break
	#Detect face in frame. faceNet is a DNN
	resultImg, faceBoxes = highlightFace(faceNet, frame)
	if not faceBoxes:
		print("No Face Detected")
	#Padding adds a buffer to the image when extracting the ROI. 
	for faceBox in faceBoxes:
		face = frame[max(0, faceBox[1]-padding): #0-Top left x coordinate, 1-Top left y coordinate. Top boundary of the ROI.
					 min(faceBox[3]+padding, frame.shape[0]-1), max(0, faceBox[0]-padding) #Bottom boundary of the ROI. -1 ensures height is not exceeded.
					 :min(faceBox[2]+padding, frame.shape[1]-1)]#Right boundary of the ROI.
		#Age and gender detection. For each detected face, the region of interest is converted into a blob then output is given.
		blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), Model_Mean_Values, swapRB=False)
		genderNet.setInput(blob)
		genderPreds = genderNet.forward()
		gender = genderList[genderPreds[0].argmax()]
		print(f"Gender: {gender}")
		ageNet.setInput(blob)
		agePreds = ageNet.forward()
		age = ageList[agePreds[0].argmax()]
		print(f"Age {age[1:-1]} years")

		#Displays the age and gender on the frame next to the detected face.
		cv2.putText(resultImg, f"{gender}, {age}", (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
		#Display results.
		cv2.imshow("Detecting Age and Gender", resultImg)
#Command line interface allows for choosing either webcam or an already existing image.
#In the command line when using webcam, run python Gender_Age_Model.py
#In the command line when using already existing image, run python gad.py --image path_to_image.jpg
