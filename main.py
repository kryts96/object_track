#   real time object detection
#   https://pjreddie.com/darknet/yolo/
#   https://www.youtube.com/watch?v=1LCb1PVqzeY

#   ip camera ptz control
#   https://stackoverflow.com/questions/54182614/control-ptz-of-an-ip-camera
#   https://github.com/FalkTannhaeuser/python-onvif-zeep

import datetime
import cv2
import imutils
import numpy as np

capture = cv2.VideoCapture("rtsp://admin:admin123@192.168.1.254:554/1")

avg = None

while(True):
    ret, frame = capture.read()
    timestamp = datetime.datetime.now()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if avg is None:
            print("[INFO] starting background model...")
            avg = gray.copy().astype("float")
            #frame.truncate(0)
            continue
        cv2.accumulateWeighted(gray, avg, 0.5)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        thresh = cv2.threshold(frameDelta, 5, 255,
                               cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 5000:
                continue

            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame2 = frame[x:x+w, y:y+h]
            cv2.imshow('test',frame2)
            #text = "Occupied"

        ## draw the text and timestamp on the frame
        #ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
        #cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        #           0.35, (0, 0, 255), 1)
        cv2.imshow('IP Camera', frame)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()