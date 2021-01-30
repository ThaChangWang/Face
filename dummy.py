import cv2
import numpy as np
import dlib
import os

def load_images_from_folder(folder):
    images = []
    if (os.path.isdir(folder)):
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
    return images

left_eyebrow_imgs = load_images_from_folder("left_eyebrow")
right_eyebrow_imgs = load_images_from_folder("right_eyebrow")
left_eye_imgs = load_images_from_folder("left_eye")
right_eye_imgs = load_images_from_folder("right_eye")
nose_imgs = load_images_from_folder("nose")
mouth_imgs = load_images_from_folder("mouth")

cap = cv2.VideoCapture("dummyvid.mp4")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

current_frame = 0

left_eyebrow_offset = 10
right_eyebrow_offset = 10
left_eye_offset = 10
right_eye_offset = 10
nose_offset = 10
mouth_offset = 10


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

        #mask = np.zeros_like(frame)

        '''

        points = []

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append([x, y])
            cv2.circle(mask, (x, y), 4, (255, 255, 255), -1)

        '''
        #left_eyebrow
        if (os.path.isdir("left_eyebrow")):
            left_eyebrow_resize = cv2.resize(left_eyebrow_imgs[current_frame % len(left_eyebrow_imgs)], ((landmarks.part(21).x + left_eyebrow_offset) - (landmarks.part(17).x - left_eyebrow_offset), (landmarks.part(17).y + left_eyebrow_offset) - (landmarks.part(19).y - left_eyebrow_offset)))

            left_eyebrow_gray = cv2.cvtColor(left_eyebrow_resize, cv2.COLOR_BGR2GRAY)

            _, left_eyebrow_mask = cv2.threshold(left_eyebrow_gray, 0, 255, cv2.THRESH_BINARY)
            left_eyebrow_mask_inv = cv2.bitwise_not(left_eyebrow_mask)

            left_eyebrow_box = frame[landmarks.part(19).y - left_eyebrow_offset:landmarks.part(17).y + left_eyebrow_offset, landmarks.part(17).x - left_eyebrow_offset:landmarks.part(21).x + left_eyebrow_offset]

            left_eyebrow_area = cv2.bitwise_and(left_eyebrow_box, left_eyebrow_box, mask=left_eyebrow_mask_inv)

            final_left_eyebrow = cv2.add(left_eyebrow_area, left_eyebrow_resize)

            frame[landmarks.part(19).y - left_eyebrow_offset:landmarks.part(17).y + left_eyebrow_offset, landmarks.part(17).x - left_eyebrow_offset:landmarks.part(21).x + left_eyebrow_offset] = final_left_eyebrow

        #right_eyebrow
        if (os.path.isdir("right_eyebrow")):
            right_eyebrow_resize = cv2.resize(right_eyebrow_imgs[current_frame % len(right_eyebrow_imgs)], ((landmarks.part(26).x + right_eyebrow_offset) - (landmarks.part(22).x - right_eyebrow_offset), (landmarks.part(26).y + right_eyebrow_offset) - (landmarks.part(24).y - right_eyebrow_offset)))

            right_eyebrow_gray = cv2.cvtColor(right_eyebrow_resize, cv2.COLOR_BGR2GRAY)

            _, right_eyebrow_mask = cv2.threshold(right_eyebrow_gray, 0, 255, cv2.THRESH_BINARY)
            right_eyebrow_mask_inv = cv2.bitwise_not(right_eyebrow_mask)

            right_eyebrow_box = frame[landmarks.part(24).y - right_eyebrow_offset:landmarks.part(26).y + right_eyebrow_offset, landmarks.part(22).x - right_eyebrow_offset:landmarks.part(26).x + right_eyebrow_offset]

            right_eyebrow_area = cv2.bitwise_and(right_eyebrow_box, right_eyebrow_box, mask=right_eyebrow_mask_inv)

            final_right_eyebrow = cv2.add(right_eyebrow_area, right_eyebrow_resize)

            frame[landmarks.part(24).y - right_eyebrow_offset:landmarks.part(26).y + right_eyebrow_offset, landmarks.part(22).x - right_eyebrow_offset:landmarks.part(26).x + right_eyebrow_offset] = final_right_eyebrow

        #left_eye
        if (os.path.isdir("left_eye")):
            left_eye_resize = cv2.resize(left_eye_imgs[current_frame % len(left_eye_imgs)], ((landmarks.part(39).x + left_eye_offset) - (landmarks.part(36).x - left_eye_offset), (landmarks.part(41).y + left_eye_offset) - (landmarks.part(37).y - left_eye_offset)))

            left_eye_gray = cv2.cvtColor(left_eye_resize, cv2.COLOR_BGR2GRAY)

            _, left_eye_mask = cv2.threshold(left_eye_gray, 0, 255, cv2.THRESH_BINARY)
            left_eye_mask_inv = cv2.bitwise_not(left_eye_mask)

            left_eye_box = frame[landmarks.part(37).y - left_eye_offset:landmarks.part(41).y + left_eye_offset, landmarks.part(36).x - left_eye_offset:landmarks.part(39).x + left_eye_offset]

            left_eye_area = cv2.bitwise_and(left_eye_box, left_eye_box, mask=left_eye_mask_inv)

            final_left_eye = cv2.add(left_eye_area, left_eye_resize)

            frame[landmarks.part(37).y - left_eye_offset:landmarks.part(41).y + left_eye_offset, landmarks.part(36).x - left_eye_offset:landmarks.part(39).x + left_eye_offset] = final_left_eye

        #right_eye
        if (os.path.isdir("right_eye")):
            right_eye_resize = cv2.resize(right_eye_imgs[current_frame % len(right_eye_imgs)], ((landmarks.part(45).x + right_eye_offset) - (landmarks.part(42).x - right_eye_offset), (landmarks.part(46).y + right_eye_offset) - (landmarks.part(44).y - right_eye_offset)))

            right_eye_gray = cv2.cvtColor(right_eye_resize, cv2.COLOR_BGR2GRAY)

            _, right_eye_mask = cv2.threshold(right_eye_gray, 0, 255, cv2.THRESH_BINARY)
            right_eye_mask_inv = cv2.bitwise_not(right_eye_mask)

            right_eye_box = frame[landmarks.part(44).y - right_eye_offset:landmarks.part(46).y + right_eye_offset, landmarks.part(42).x - right_eye_offset:landmarks.part(45).x + right_eye_offset]

            right_eye_area = cv2.bitwise_and(right_eye_box, right_eye_box, mask=right_eye_mask_inv)

            final_right_eye = cv2.add(right_eye_area, right_eye_resize)

            frame[landmarks.part(44).y - right_eye_offset:landmarks.part(46).y + right_eye_offset, landmarks.part(42).x - right_eye_offset:landmarks.part(45).x + right_eye_offset] = final_right_eye

        #nose
        if (os.path.isdir("nose")):
            nose_resize = cv2.resize(nose_imgs[current_frame % len(nose_imgs)], ((landmarks.part(35).x + nose_offset) - (landmarks.part(31).x - nose_offset), (landmarks.part(33).y + nose_offset) - (landmarks.part(27).y - nose_offset)))

            nose_gray = cv2.cvtColor(nose_resize, cv2.COLOR_BGR2GRAY)

            _, nose_mask = cv2.threshold(nose_gray, 0, 255, cv2.THRESH_BINARY)
            nose_mask_inv = cv2.bitwise_not(nose_mask)

            nose_box = frame[landmarks.part(27).y - nose_offset:landmarks.part(33).y + nose_offset, landmarks.part(31).x - nose_offset:landmarks.part(35).x + nose_offset]

            nose_area = cv2.bitwise_and(nose_box, nose_box, mask=nose_mask_inv)

            final_nose = cv2.add(nose_area, nose_resize)

            frame[landmarks.part(27).y - nose_offset:landmarks.part(33).y + nose_offset, landmarks.part(31).x - nose_offset:landmarks.part(35).x + nose_offset] = final_nose

        #mouth
        if (os.path.isdir("mouth")):
            mouth_resize = cv2.resize(mouth_imgs[current_frame % len(mouth_imgs)], ((landmarks.part(54).x + mouth_offset) - (landmarks.part(48).x - mouth_offset), (max(landmarks.part(58).y, landmarks.part(56).y) + mouth_offset) - (min(landmarks.part(50).y, landmarks.part(52).y) - mouth_offset)))

            mouth_gray = cv2.cvtColor(mouth_resize, cv2.COLOR_BGR2GRAY)

            _, mouth_mask = cv2.threshold(mouth_gray, 0, 255, cv2.THRESH_BINARY)
            mouth_mask_inv = cv2.bitwise_not(mouth_mask)

            mouth_box = frame[(min(landmarks.part(50).y, landmarks.part(52).y) - mouth_offset):(max(landmarks.part(58).y, landmarks.part(56).y) + mouth_offset), (landmarks.part(48).x - mouth_offset):(landmarks.part(54).x + mouth_offset)]

            mouth_area = cv2.bitwise_and(mouth_box, mouth_box, mask=mouth_mask_inv)

            final_mouth = cv2.add(mouth_area, mouth_resize)

            frame[(min(landmarks.part(50).y, landmarks.part(52).y) - mouth_offset):(max(landmarks.part(58).y, landmarks.part(56).y) + mouth_offset), (landmarks.part(48).x - mouth_offset):(landmarks.part(54).x + mouth_offset)] = final_mouth

        




        cv2.imshow("frame", frame)

        current_frame += 1
        print(current_frame)



    key = cv2.waitKey(1)
    if key == 27:
        break

