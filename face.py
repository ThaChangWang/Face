import cv2
import numpy as np
import dlib
import os
import sys


cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

current_frame = 0

os.system("mkdir " + sys.argv[1])

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_copy = frame.copy()

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
            #cv2.circle(mask, (x, y), 1, (255, 255, 255), -1)

        points = np.array(points)


        #face_points = np.append(points[0:16], np.flipud(points[17:26]), axis=0)

        if (sys.argv[1] == "left_eyebrow"):

            left_eyebrow_points = points[17:22]

            mask = cv2.fillPoly(frame, [left_eyebrow_points], (0, 0, 0))

            frame = cv2.bitwise_xor(frame_copy, mask)

            area = frame[landmarks.part(19).y:landmarks.part(17).y, landmarks.part(17).x:landmarks.part(21).x]

        elif (sys.argv[1] == "right_eyebrow"):

            right_eyebrow_points = points[22:27]

            mask = cv2.fillPoly(frame, [right_eyebrow_points], (0, 0, 0))

            frame = cv2.bitwise_xor(frame_copy, mask)

            area = frame[landmarks.part(24).y:landmarks.part(26).y, landmarks.part(22).x:landmarks.part(26).x]

        elif (sys.argv[1] == "left_eye"):

            left_eye_points = points[36:42]

            mask = cv2.fillPoly(frame, [left_eye_points], (0, 0, 0))

            frame = cv2.bitwise_xor(frame_copy, mask)

            area = frame[landmarks.part(37).y:landmarks.part(41).y, landmarks.part(36).x:landmarks.part(39).x]

        elif (sys.argv[1] == "right_eye"):

            right_eye_points = points[42:48]

            mask = cv2.fillPoly(frame, [right_eye_points], (0, 0, 0))

            frame = cv2.bitwise_xor(frame_copy, mask)

            area = frame[landmarks.part(44).y:landmarks.part(46).y, landmarks.part(42).x:landmarks.part(45).x]

        elif (sys.argv[1] == "nose"):

            nose_points = np.array([points[27], points[31], points[33], points[35]])

            mask = cv2.fillPoly(frame, [nose_points], (0, 0, 0))

            frame = cv2.bitwise_xor(frame_copy, mask)

            area = frame[landmarks.part(27).y:landmarks.part(33).y, landmarks.part(31).x:landmarks.part(35).x]


        elif (sys.argv[1] == "mouth"):

            mouth_points = points[48:60]

            mask = cv2.fillPoly(frame, [mouth_points], (0, 0, 0))

            frame = cv2.bitwise_xor(frame_copy, mask)

            area = frame[min(landmarks.part(50).y, landmarks.part(52).y):max(landmarks.part(58).y, landmarks.part(56).y), landmarks.part(48).x:landmarks.part(54).x]



        #cv2.imwrite("mask/mask" + str(current_frame) + ".png", mask_crop)

        cv2.imwrite(sys.argv[1] + "/" + sys.argv[1] + str(current_frame) + ".png", area)

        #current_frame += 1


        cv2.imshow("Frame", frame)

    current_frame += 1


    key = cv2.waitKey(1)
    if key == 27:
        break

