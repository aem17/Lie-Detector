from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import datetime

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    mA = dist.euclidean(mouth[13], mouth[19])
    mB = dist.euclidean(mouth[14], mouth[18])
    mC = dist.euclidean(mouth[15], mouth[17])
    # mD = dist.euclidean(mouth[4], mouth[8])
    # mE = dist.euclidean(mouth[5], mouth[7])
    mF = dist.euclidean(mouth[12], mouth[16])

    mar = (mA + mB + mC) / (3.0 * mF)
    return mar

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
                help="path to input video file")
args = vars(ap.parse_args())

MOUTH_AR_THRESH = 0.05
EYE_AR_THRESH = 0.31
MOUTH_AR_CONSEC_FRAMES = 3
EYE_AR_CONSEC_FRAMES = 3

frameCOUNTER = 0
mCOUNTER = 0
eyeCOUNTER = 0
mouthCOUNTER = 0
TOTAL = 0
mTOTAL = 0
Blinkrate = 0

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

if not args.get("video", False):
    vs = VideoStream(src=0).start()
    fileStream = False
else:
    vs = FileVideoStream(args["video"]).start()
    fileStream = True
time.sleep(1.0)

earList = []
while True:
    if fileStream and not vs.more():
        break
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)


    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        mouth = shape[mStart:mEnd]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        mar = mouth_aspect_ratio(mouth)
        ear = (leftEAR + rightEAR) / 2.0

        if frameCOUNTER <= 20:
            frameCOUNTER += 1
        else:
            frameCOUNTER = 0
            frameCOUNTER += 1
            earList.pop(0)
            
        earList.append(ear)
        earAverage = sum(earList) / len(earList)

        mouthHull = cv2.convexHull(mouth)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if earAverage < EYE_AR_THRESH:
            eyeCOUNTER += 1
            cv2.putText(frame, "Blinkrate: {:.2f}".format(Blinkrate), (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if eyeCOUNTER == 1:
                t1 = datetime.datetime.now()

        else:
            cv2.putText(frame, "Blinkrate: {:.2f}".format(Blinkrate), (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if eyeCOUNTER >= EYE_AR_CONSEC_FRAMES:
                t2 = datetime.datetime.now()
                TOTAL += 1

                timeDiff = t2 - t1
                Blinkrate = (TOTAL / timeDiff.seconds) if timeDiff.seconds != 0 else 0

                eyeCOUNTER = 0

        if mar < MOUTH_AR_THRESH:
            mouthCOUNTER += 1

        elif mar >= MOUTH_AR_THRESH:
            mouthCOUNTER += 1
            if Blinkrate >= 1.25 * Blinkrate:
                # print("Lie")
                cv2.putText(frame, 'Lie', (300, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # print("Truth")
                cv2.putText(frame, 'Truth', (300, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(earAverage), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
