from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str, help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1, help="Flag to display the output frame on screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn", help="face detection model to use: 'hog' or 'cnn'")
args = vars(ap.parse_args())

print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

print("[INFO] starting the video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

count=0
while True:
    frame = vs.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])


    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])



    for (top,right,bottom,left) in boxes:

        top = int(top*r)
        right = int(right*r)
        bottom = int(bottom*r)
        left = int(left*r)

        cv2.rectangle(frame, (left,top), (right,bottom), (0,255,0), 2)
        y = top - 15 if top-15 > 15 else top + 15

        cv2.imwrite("collectedfaces/face" + str(count) + ".jpg", rgb)

        count+=1
        print(count)

    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key==ord("q"):
            break
    if count >= 150: # stop video after reaching sample thresh
        #print('inbreak')
        break

cv2.destroyAllWindows()
vs.stop()

if writer is not None:
    writer.release()



