import cv2
import mediapipe as mp
import time
ptime = 0
cap = cv2.VideoCapture(0)
myFace = mp.solutions.face_detection
face = myFace.FaceDetection()
myDraw = mp.solutions.drawing_utils
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = face.process(imgRGB)
    if result.detections:
        for id,detection in enumerate(result.detections):
            # myDraw.draw_detection(img,detection)
            bboxC = detection.location_data.relative_bounding_box
            ih,iw,ic = img.shape
            bbox = int(bboxC.xmin*iw),int(bboxC.ymin*ih), \
                int(bboxC.width*iw),int(bboxC.height*ih)
            cv2.rectangle(img,bbox,color=(255,255,0),thickness=3)
            cv2.putText(img,f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_COMPLEX,3,(255,255,0),3)
    ctime = time.time()
    fps =1/(ctime-ptime)
    ptime=ctime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_DUPLEX,3,(0,0,0),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)