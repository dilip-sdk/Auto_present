import os
from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np

# Define the dimensions
width, height = 1280, 720  # Assuming imgCurrent.shape is (1080, 1920, 3)
folderPath = "Presentation"
imgNumber = 0
hs, ws = int(120 * 1), int(213 * 1)  # width and height of small image
gestureThreshold =300

# Initialize the webcam
cap = cv2.VideoCapture(0)
# cap= cv2.VideoCapture("http://192.168.20.4:8080/video")
cap.set(3, width)
cap.set(4, height)
buttonPressed = False
counter = 0
delay = 30
annotations = [[]]
annotationNumber = 0
annotationStart = False

# Load and sort images from the folder
pathImages = sorted(os.listdir(folderPath), key=len)

# Hand Detector
detectorHand = HandDetector(detectionCon=0.9, maxHands=1)

# Initialize variables for smoothing filter
smoothening = 2
prev_x, prev_y = 0, 0
while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam")
        break

    img = cv2.flip(img, 1)

    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)
    # print(imgCurrent.shape)
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)
    hands, img = detectorHand.findHands(img)  # with draw
    if hands and buttonPressed is False:
        hand = hands[0]
        cx, cy = hand["center"]
        lmList = hand["lmList"]  # List of 21 Landmark points
        fingers = detectorHand.fingersUp(hand)
        xVal = int(np.interp(lmList[8][0], [(width // 2), width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))
        xVal = prev_x + (xVal - prev_x) // smoothening
        yVal = prev_y + (yVal - prev_y) // smoothening
        prev_x, prev_y = xVal, yVal
        indexFinger = xVal, yVal
        # print(fingers)

        if cy <= gestureThreshold:  # If hand is at the height of the face
            if fingers == [1, 0, 0, 0, 0]:
                # print("Left")

                if imgNumber>0:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = 0
                    annotationStart = False
                    imgNumber-=1
            if fingers == [0, 0, 0, 0, 1]:
                # print("Right")

                if imgNumber <len(pathImages)-1:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = 0
                    annotationStart = False
                    imgNumber += 1
            #Gesture 3-show Pointer
        if fingers==[0,1,1,0,0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            print(annotationNumber)
            annotations[annotationNumber].append(indexFinger)
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        else:
            annotationStart = False
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True
        if fingers == [1, 1, 1, 1, 1]:
            annotationNumber =0
            annotations = [[]]
            buttonPressed = True

    if buttonPressed:
        counter += 1
        if counter > delay:
            counter = 0
            buttonPressed = False

    for i, annotation in enumerate(annotations):
        for j in range(len(annotation)):
            if j != 0:
                cv2.line(imgCurrent, annotation[j - 1], annotation[j], (0, 0, 200), 12)

    h, w, _ = imgCurrent.shape

    # Resize imgSmall with the adjusted width (ws)
    imgSmall = cv2.resize(img, (ws, hs))

    # Define the position for the small image (top right corner, moved left by 10 pixels)
    x_offset = w - ws -400  # Adjust this value to move further left or right
    y_offset = 0

    # Ensure the region we're copying to is valid
    if y_offset + hs <= h and x_offset >= 0 and x_offset + ws <= w:
        imgCurrent[y_offset:y_offset + hs, x_offset:x_offset + ws] = imgSmall
    else:
        print("Image dimensions are not sufficient for the small image overlay")
    cv2.imshow("Slides", imgCurrent)
    # cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
