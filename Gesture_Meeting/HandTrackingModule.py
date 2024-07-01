import cv2
import mediapipe as mp
import time
class handDectector():
    def __init__(self,mode=False,max_hands=2,model_complexity=1,detectionconfidence=0.5,trackingconfidence=0.5):
        self.mode=mode
        self.maxHands=max_hands
        self.model_com=model_complexity
        self.detectionCon=detectionconfidence
        self.trackCon=trackingconfidence
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode,self.maxHands,self.model_com,self.detectionCon,self.trackCon)
        self.mpdraw = mp.solutions.drawing_utils


    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, handLms, self.mphands.HAND_CONNECTIONS)

        return  img
    def findPosition(self,img,handNo=0,draw=True):
        lmList=[]
        if self.results.multi_hand_landmarks:
            myhand=self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myhand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = (int(lm.x * w), int(lm.y * h))
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
              #if id == 0:
               #     cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
                #if id == 1:

        return lmList
def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    detector=handDectector()
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img=detector.findHands(img)
        lmList=detector.findPosition(img)
        if len(lmList)!=0:
            print(lmList[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Wait for 'q' key to be pressed
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__=="__main__":
    main()