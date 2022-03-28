#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : Anirudh Kakarlapudi
# References: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

import cv2


def image_face_detect(image, number):
    """ Detects the faces inside the image with a pretrained cascade
    classfier and displays the image

    Args:
        image(array)
    """
    clf_path = cv2.data.haarcascades+'/haarcascade_frontalface_default.xml'
    face_clf = cv2.CascadeClassifier(clf_path)
    if face_clf.empty():
        raise Exception("Classifier not loaded successfully. Check the path")

    # Convert the image into grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Equilizes the histogram of a grayscale image
    img_gray = cv2.equalizeHist(img_gray)

    # Detect the faces
    faces = face_clf.detectMultiScale(img_gray,
                                      scaleFactor=1.5,
                                      minNeighbors=5)
    for (x, y, w, h) in faces:
        # Draw the box around the faces
        image = cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
        image = cv2.putText(image, str(number), (x-1,y-4),
                            0, 1, (255, 255, 0), 1, 1)

    cv2.imshow("Video with Face Detection (Press q/ESC to quit)", image)
    return image


def video_face_detect():
    """ Breaks the live video into frames and finds faces in each frame and
    displays them
    """
    video_capture = cv2.VideoCapture(0)
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    size = (frame_width, frame_height)
    final = cv2.VideoWriter("face_detection.avi", cv2.VideoWriter_fourcc(*'MJPG'),
                             fps=30, frameSize=size)
    frame_count = 0
    while True:
        _, frame = video_capture.read()
        frame_count += 1
        img = image_face_detect(frame, frame_count)
        final.write(img)
        k = cv2.waitKey(10)
        # Exit when ESC or q is pressed
        if k == 27 or k == ord('q'):
            break
    # Close all the windows at the end
    video_capture.release()
    final.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_face_detect()
