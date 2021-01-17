import cv2
import numpy as np
import dlib
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

masks = load_images_from_folder("mask")
faces = load_images_from_folder("face")


cap = cv2.VideoCapture("mitch.mp4")

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

        dummy_mask = np.zeros_like(frame)

        points = []

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append([x, y])
            cv2.circle(dummy_mask, (x, y), 4, (255, 255, 255), -1)

        best_mask = ""
        best_val = 0
        best_loc = (0, 0)

        for mask in masks:

            width = mask.shape[1]
            width_factor = width//20
            height = mask.shape[0]
            height_factor = height//20

            while(width > 360):

            mask = cv2.resize(mask, (width, height))
            
            result = cv.matchTemplate(dummy_mask, mask, cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            print(max_val)

            if max_val >= best_val:
                best_val = max_val
                best_loc = max_loc

            width -= width_factor
            height -= height_factor

        



        points = np.array(points)

        face_points = np.append(points[0:16], np.flipud(points[17:26]), axis=0)

        #mask = cv2.fillPoly(mask, [face_points], (255, 255, 255))
        frame = cv2.bitwise_and(frame, mask)

        current_frame += 1



    key = cv2.waitKey(1)
    if key == 27:
        break

