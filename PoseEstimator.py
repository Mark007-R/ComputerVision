import cv2
import mediapipe as mp
import time

class PoseEstimator:
    def __init__(self,mode=False,modelComp=1,smoothLandmarks=True,enableSeg=False,smoothSeg=True,detectionCon=0.5,trackingCon=0.5):
        self.mode = mode
        self.modelComp = modelComp
        self.smoothLandmarks = smoothLandmarks
        self.enableSeg = enableSeg
        self.smoothSeg = smoothSeg
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.myPose = mp.solutions.pose
        self.pose = self.myPose.Pose(self.mode,self.modelComp,self.smoothLandmarks,self.enableSeg,self.smoothSeg,self.detectionCon,self.trackingCon)
        self.myDraw = mp.solutions.drawing_utils
    
    def poseest(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result = self.pose.process(imgRGB)
        if draw:
            if self.result.pose_landmarks:
                self.myDraw.draw_landmarks(img,self.result.pose_landmarks,self.myPose.POSE_CONNECTIONS)
        return img
    def circlePoint(self,img,val):
        lmlist = []
        if self.result.pose_landmarks:
            for id,lm in enumerate(self.result.pose_landmarks.landmark):
                    h ,w, c = img.shape
                    cx,cy = int(lm.x*w),int(lm.y*h)
                    lmlist.append([id,cx,cy])
                    if(id == val):
                        cv2.circle(img,(cx,cy),7,(0,0,0),cv2.FILLED)
        return lmlist,img

def main():
    ptime = 0
    cap = cv2.VideoCapture(1)
    p_es = PoseEstimator()
    while True:
        success,img = cap.read()
        img = p_es.poseest(img)
        val,img = p_es.circlePoint(img,0)

        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(255,255,255),3)
        img = cv2.resize(img,(1000,600))
        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()