import cv2

print(cv2.data.haarcascades)

face_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

webcam = cv2.VideoCapture(0)

while True:

	successCallback, frame = webcam.read()

	if not successCallback:
		break

	frameToGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_model.detectMultiScale(frameToGray, scaleFactor=1.7)
	print(faces)

	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)

	cv2.imshow("test", frame)

	cv2.waitKey(1)

#cleanup
webcam.release()
cv2.destroyAllWindows()
print("squeaky clean")