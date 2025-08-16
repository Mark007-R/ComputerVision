import cv2
import mediapipe as mp
import time
class FaceDetector:
    def __init__(self,detectionCon=0.5, model_selection=0):
        self.detectionCon = detectionCon
        self.modelselection = model_selection
        self.myFace = mp.solutions.face_detection
        self.face = self.myFace.FaceDetection(self.detectionCon,self.modelselection)
        self.myDraw = mp.solutions.drawing_utils

    def faceDetect(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        result = self.face.process(imgRGB)
        bboxs = []
        if result.detections:
            for id,detection in enumerate(result.detections):
                # myDraw.draw_detection(img,detection)
                bboxC = detection.location_data.relative_bounding_box
                ih,iw,ic = img.shape
                bbox = int(bboxC.xmin*iw),int(bboxC.ymin*ih), \
                       int(bboxC.width*iw),int(bboxC.height*ih)
                bboxs.append([id,bbox,detection.score])
                if draw:
                    img = self.bestborder(img,bbox)
                    cv2.putText(img,f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_COMPLEX,3,(255,255,0),2)
        return img
    def bestborder(self,img,bbox,l=35):
        x,y,w,h = bbox
        x1,y1=x+w,y+h
        cv2.rectangle(img,bbox,color=(255,255,0),thickness=2)        
        cv2.line(img,(x,y),(x+l,y),(255,255,0),7)
        cv2.line(img,(x,y),(x,y+l),(255,255,0),7)

        cv2.line(img,(x1,y),(x-l,y),(255,255,0),7)
        cv2.line(img,(x1,y),(x,y+l),(255,255,0),7)

        cv2.line(img,(x,y1),(x+l,y),(255,255,0),7)
        cv2.line(img,(x,y1),(x,y-l),(255,255,0),7)

        cv2.line(img,(x1,y),(x-l,y),(255,255,0),7)
        cv2.line(img,(x1,y),(x,y-l),(255,255,0),7)

        return img

def main():
    ptime = 0
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    fd = FaceDetector()
    while True:
        success, img = cap.read()
        img = fd.faceDetect(img) 
        ctime = time.time()
        fps =1/(ctime-ptime)
        ptime=ctime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_DUPLEX,3,(0,0,0),3)
        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()