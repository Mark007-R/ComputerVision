import cv2
import mediapipe as mp
import time

class handDetector:

    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands 
        self.detectionCon = detectionCon
        self.tracCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findhands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result1 = self.hands.process(imgRGB)
        if self.result1.multi_hand_landmarks:
            for handlms in self.result1.multi_hand_landmarks:
                if draw == True:
                    self.mpDraw.draw_landmarks(img,handlms,self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findposition(self,img,idx,handNo=0,draw=True):
        lmList = []
        if self.result1.multi_hand_landmarks:
            myHand = self.result1.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                h ,w, c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    if(id == idx):
                        cv2.circle(img,(cx,cy),7,(0,0,0),cv2.FILLED)
        return lmList,img

class faceDetector:

    def __init__(self,mode=False,max_faces=1,refine_landmarks=False,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.max_faces = max_faces
        self.refineLandmarks = refine_landmarks
        self.mode = detectionCon
        self.trackCon = trackCon
        self.myFace = mp.solutions.face_mesh
        self.face = self.myFace.FaceMesh()
        self.myDraw = mp.solutions.drawing_utils
    
    def findnose(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result2 = self.face.process(imgRGB)
        if self.result2.multi_face_landmarks:
            for handlms in self.result2.multi_face_landmarks:
                if draw:
                    self.myDraw.draw_landmarks(img,handlms,self.myFace.FACEMESH_NOSE)
        return img

    def findcontours(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result3 = self.face.process(imgRGB)
        if self.result3.multi_face_landmarks:
            for handlms in self.result3.multi_face_landmarks:
                if draw:
                    self.myDraw.draw_landmarks(img,handlms,self.myFace.FACEMESH_CONTOURS)
        return img 
    
    def findposition(self,img,idx,draw=True):
        lmList = []
        if self.result3.multi_face_landmarks:
            for myFace in self.result3.multi_face_landmarks:
                for id,lm in enumerate(myFace.landmark):
                    h ,w, c = img.shape
                    cx,cy = int(lm.x*w),int(lm.y*h)
                    lmList.append([id,cx,cy])
                    if draw:
                        if(id == idx):
                            cv2.circle(img,(cx,cy),7,(0,0,0),cv2.FILLED)
        return lmList,img

def main():
    cap = cv2.VideoCapture(1)
    pTime = 0
    cTime = 0
    detectorh = handDetector()
    detectorf = faceDetector()
    while True:
        success,img = cap.read()
        img = detectorh.findhands(img)
        val,img = detectorh.findposition(img,8)

        img = detectorf.findcontours(img)
        img = detectorf.findnose(img)
        val,img = detectorf.findposition(img,2)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(255,255,255),3)

        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()