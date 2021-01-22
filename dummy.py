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

maskImgs = load_images_from_folder("mask")
faceImgs = load_images_from_folder("face")


cap = cv2.VideoCapture("dummyvid.mp4")

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

        face_points = np.append(points[0:16], np.flipud(points[17:26]), axis=0)

        best_mask = 0
        best_val = 0
        best_loc = (0, 0)
        #best_width = 0
        #best_height = 0

        current_mask = 0

        for mask in maskImgs:

            #width = mask.shape[1]
            #width_factor = width//10
            #height = mask.shape[0]
            #height_factor = height//10

            #while(width > 360):

            mask = cv2.resize(mask, (x2-x1, y2-y1))
                
            result = cv2.matchTemplate(dummy_mask, mask, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            

            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_mask = current_mask
                #best_width = width
                #best_height = height

            #width -= width_factor
            #height -= height_factor

            current_mask += 1


        #best_fits.write(best_mask + " " + str(best_val) + " " + str(best_loc) + "\n")


        #points = np.array(points)

        #face_points = np.append(points[0:16], np.flipud(points[17:26]), axis=0)

        #mask = cv2.fillPoly(mask, [face_points], (255, 255, 255))

        best_face_img = faceImgs[best_mask]

        best_face_resize = cv2.resize(best_face_img, (x2-x1, y2-y1))

        #best_face_gray = cv2.cvtColor(best_face_resize, cv2.COLOR_BGR2GRAY)

        #_, face_mask = cv2.threshold(best_face_gray, 0, 255, cv2.THRESH_BINARY_INV)

        face_area = frame[y1:y2, x1:x2]

        no_face = cv2.fillPoly(frame, [face_points], (0, 0, 0))

        final_frame = cv2.add(no_face[y1:y2, x1:x2], best_face_resize)      



        cv2.imshow("frame", final_frame)

        current_frame += 1
        print(current_frame)



    key = cv2.waitKey(1)
    if key == 27:
        break

