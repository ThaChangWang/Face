import cv2
import numpy as np
import dlib
import os

os.system("mkdir face")
os.system("mkdir mask")

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

current_frame = 0

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)

        mask = np.zeros_like(frame)

        points = []

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append([x, y])
            cv2.circle(mask, (x, y), 4, (255, 255, 255), -1)

        points = np.array(points)

        face_points = np.append(points[0:16], np.flipud(points[17:26]), axis=0)

        cv2.imwrite("mask/mask" + str(current_frame) + ".png", mask)

        mask = cv2.fillPoly(mask, [face_points], (255, 255, 255))
        frame = cv2.bitwise_and(frame, mask)

        cv2.imwrite("face/face" + str(current_frame) + ".png", frame)

        current_frame += 1


    cv2.imshow("Frame", frame)


    key = cv2.waitKey(1)
    if key == 27:
        break

