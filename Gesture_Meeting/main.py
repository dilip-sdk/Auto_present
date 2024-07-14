from flask import Flask, render_template, Response
import cv2
import os
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from thickness import GestureThicknessController
from slides import save_document_pages_as_images
from draw import Show
from thickness import GestureThicknessController
app = Flask(__name__)

class GesturePresentation:
    def __init__(self,folderP, detection=True, doc_path=""):
        self.detection = detection
        self.doc_path = doc_path
        self.folderPath = folderP
        self.brushThickness = 12
        self.eraserThickness = 20
        self.controller = GestureThicknessController()
        self.width, self.height = 1280, 720
        self.gestureThreshold = 400
        self.buttonPressed = False
        self.counter = 0
        self.delay = 30
        self.annotations = [[]]
        self.annotationColors = [(255, 0, 255)]
        self.annotationThicknesses = [self.brushThickness]
        self.annotationNumber = 0
        self.annotationStart = False
        self.smoothening = 5
        self.prev_x, self.prev_y = 0, 0
        self.drawColor = (255, 0, 0)
        self.Mode = "brush"
        self.thicknessModeActive = False
        self.imgNumber = 0

        if len(self.doc_path) != 0:
            save_document_pages_as_images(self.doc_path, self.folderPath)

        self.pathImages = sorted(os.listdir(self.folderPath), key=len)
        print(f"Loaded images: {self.pathImages}")

        self.detectorHand = HandDetector(detectionCon=0.9, maxHands=1)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.width)
        self.cap.set(4, self.height)

        self.overlayList1 = self.load_images("Header")
        self.header = self.overlayList1[0]
        self.overlayList2 = self.load_images("thickness_image")
        self.header_brush = self.overlayList2[0]
        self.header_erase = self.overlayList2[1]


    def load_images(self, folderPath):
        images = []
        myList = os.listdir(folderPath)
        for imPath in myList:
            image = cv2.imread(f'{folderPath}/{imPath}')
            if image is not None:
                images.append(image)
            else:
                print(f"Failed to load image: {imPath}")
        return images
        # Your existing run method

    def run(self):
        if len(self.pathImages) != 0:
            while True:
                success, img = self.cap.read()
                if not success:
                    print("Failed to capture image from webcam")
                    break
                img = cv2.flip(img, 1)
                imgCopy = self.process_frame(img)
                cv2.imshow("Slides", imgCopy)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            self.cap.release()
            cv2.destroyAllWindows()
            return imgCopy
        else:
            while True:
                img = Show()
                # cv2.imshow("Output", img)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            self.cap.release()
            cv2.destroyAllWindows()
            return img

    def process_frame(self, img):
        img2 = img.copy()
        img3 = img.copy()
        imgCopy = img.copy()  # Ensure imgCopy is initialized
        if self.imgNumber < len(self.pathImages):
            pathFullImage = os.path.join(self.folderPath, self.pathImages[self.imgNumber])
            imgCurrent = cv2.imread(pathFullImage)
            if imgCurrent is None:
                print(f"Failed to load image: {pathFullImage}")
                return img

            height2, width2 = imgCurrent.shape[:2]
            new_width = width2 // 2
            new_height = height2 // 2
            imgCurrent = cv2.resize(imgCurrent, (new_width, new_height))
            imgCopy = imgCurrent.copy()

            header_width = new_width // 2
            header = cv2.resize(self.header, (header_width, 50))
            header_brush = cv2.resize(self.header_brush, (80, 80))
            header_erase = cv2.resize(self.header_erase, (80, 80))
            cv2.line(img, (0, self.gestureThreshold), (self.width, self.gestureThreshold), (0, 255, 0), 10)

            if self.thicknessModeActive:
                imgCopy, self.brushThickness, self.thicknessModeActive = self.controller.process_frame(img2, self.Mode,
                                                                                                       imgCopy.copy())
                self.brushThickness = int(self.brushThickness)
                if self.brushThickness <= 0:
                    self.brushThickness = 1

            hands, img = self.detectorHand.findHands(img)
            if hands and not self.buttonPressed:
                hand = hands[0]
                cx, cy = hand["center"]
                lmList = hand["lmList"]
                fingers = self.detectorHand.fingersUp(hand)
                xVal = int(np.interp(lmList[8][0], [(self.width // 2), self.width], [0, self.width]))
                yVal = int(np.interp(lmList[8][1], [150, self.height - 150], [0, self.height]))
                xVal = self.prev_x + (xVal - self.prev_x) // self.smoothening
                yVal = self.prev_y + (yVal - self.prev_y) // self.smoothening
                self.prev_x, self.prev_y = xVal, yVal
                indexFinger = xVal, yVal

                if cy <= self.gestureThreshold:
                    if fingers == [1, 0, 0, 0, 0]:
                        if self.imgNumber > 0:
                            self.buttonPressed = True
                            self.reset_annotations()
                            self.imgNumber -= 1
                    if fingers == [0, 0, 0, 0, 1]:
                        if self.imgNumber < len(self.pathImages) - 1:
                            self.buttonPressed = True
                            self.reset_annotations()
                            self.imgNumber += 1

                if fingers == [0, 1, 1, 0, 0]:
                    if yVal < 50:
                        self.change_color(indexFinger, xVal)
                    self.activate_thickness_mode(xVal, yVal)

                if not self.thicknessModeActive:
                    cv2.circle(imgCopy, indexFinger, 12, (0, 0, 255), cv2.FILLED)

                if fingers == [0, 1, 0, 0, 0]:
                    self.draw_or_erase(indexFinger, imgCopy)
                else:
                    self.annotationStart = False

                if fingers == [0, 1, 1, 1, 0]:
                    if self.annotations:
                        self.annotations.pop(-1)
                        self.annotationColors.pop(-1)
                        self.annotationThicknesses.pop(-1)
                        self.annotationNumber -= 1
                        self.buttonPressed = True

                if fingers == [1, 1, 1, 1, 1]:
                    self.reset_annotations()
                    self.buttonPressed = True

            if self.buttonPressed:
                self.counter += 1
                if self.counter > self.delay:
                    self.counter = 0
                    self.buttonPressed = False

            self.draw_annotations(imgCopy)

            imgSmall = cv2.resize(img if self.detection else img3, (int(213 * 1), int(120 * 1)))
            self.overlay_images(imgCopy, imgSmall, header, header_brush, header_erase)
        else:
            print(f"No images to display. Image number: {self.imgNumber}")

        return imgCopy

    def reset_annotations(self):
        self.annotations = [[]]
        self.annotationColors = [self.drawColor]
        self.annotationThicknesses = [self.brushThickness]
        self.annotationNumber = 0
        self.annotationStart = False

    def change_color(self, indexFinger, xVal):
        if 360 < xVal < 410:
            self.header = self.overlayList1[0]
            self.drawColor = (255, 0, 255)
        elif 460 < xVal < 510:
            self.header = self.overlayList1[1]
            self.drawColor = (255, 0, 0)
        elif 560 < xVal < 610:
            self.header = self.overlayList1[2]
            self.drawColor = (0, 255, 0)
        elif 660 < xVal < 710:
            self.header = self.overlayList1[3]
            self.drawColor = (0, 0, 0)

    def activate_thickness_mode(self, xVal, yVal):
        if xVal < 80 and 230 < yVal < 300:
            self.thicknessModeActive = True
            self.Mode = "brush"
            self.brushThickness = 12
            self.controller.reset()
        elif xVal > 875 and 230 < yVal < 300:
            self.thicknessModeActive = True
            self.Mode = "eraser"
            self.eraserThickness = 20
            self.controller.reset()
        else:
            self.thicknessModeActive = False

    def draw_or_erase(self, indexFinger, imgCopy):
        if self.drawColor == (0, 0, 0):
            if not self.annotationStart:
                self.annotationStart = True
                self.annotationNumber += 1
                self.annotations.append([])
                self.annotationColors.append(self.drawColor)
                self.annotationThicknesses.append(self.eraserThickness)
            self.annotations[self.annotationNumber].append(indexFinger)
            cv2.circle(imgCopy, indexFinger, self.eraserThickness, self.drawColor, cv2.FILLED)
        else:
            if not self.annotationStart:
                self.annotationStart = True
                self.annotationNumber += 1
                self.annotations.append([])
                self.annotationColors.append(self.drawColor)
                self.annotationThicknesses.append(self.brushThickness)
            self.annotations[self.annotationNumber].append(indexFinger)
            cv2.circle(imgCopy, indexFinger, self.brushThickness, self.drawColor, cv2.FILLED)

    def draw_annotations(self, imgCopy):
        for i, annotation in enumerate(self.annotations):
            for j in range(1, len(annotation)):
                if annotation and self.annotationColors[i] and self.annotationThicknesses[i]:
                    cv2.line(imgCopy, annotation[j - 1], annotation[j], self.annotationColors[i],
                             self.annotationThicknesses[i])

    def overlay_images(self, imgCopy, imgSmall, header, header_brush, header_erase):
        h, w, _ = imgCopy.shape
        x_offset = w - imgSmall.shape[1]
        y_offset = h - imgSmall.shape[0]
        if y_offset >= 0 and y_offset + imgSmall.shape[0] <= h and x_offset >= 0 and x_offset + imgSmall.shape[1] <= w:
            imgCopy[y_offset:y_offset + imgSmall.shape[0], x_offset:x_offset + imgSmall.shape[1]] = imgSmall

        new_width = header.shape[1]
        x_offset2 = (w - new_width) // 2
        y_offset2 = 0
        if y_offset2 >= 0 and y_offset2 + header.shape[0] <= imgCopy.shape[
            0] and x_offset2 >= 0 and x_offset2 + new_width <= w:
            imgCopy[y_offset2:y_offset2 + header.shape[0], x_offset2:x_offset2 + new_width] = header

        center_y_offset = (imgCopy.shape[0] - header_brush.shape[0]) // 2
        if center_y_offset >= 0 and center_y_offset + header_brush.shape[0] <= imgCopy.shape[0] and 0 + \
                header_brush.shape[1] <= imgCopy.shape[1]:
            imgCopy[center_y_offset:center_y_offset + header_brush.shape[0], 0:header_brush.shape[1]] = header_brush

        right_x_offset = imgCopy.shape[1] - header_brush.shape[1]
        if center_y_offset >= 0 and center_y_offset + header_brush.shape[0] <= imgCopy.shape[
            0] and right_x_offset >= 0 and right_x_offset + header_brush.shape[1] <= imgCopy.shape[1]:
            imgCopy[center_y_offset:center_y_offset + header_brush.shape[0],
            right_x_offset:right_x_offset + header_brush.shape[1]] = header_erase

    def show(self):
        img = cv2.imread("images/sheet.jpg")
        cv2.imshow("Output", img)
        return img

    def get_frame(self):
        while True:
            success, img = self.cap.read()
            if not success:
                print("Failed to capture image from webcam")
                break
            img = cv2.flip(img, 1)
            imgCopy = self.process_frame(img)
            ret, jpeg = cv2.imencode('.jpg', imgCopy)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # Create a simple HTML template to display the video feed

@app.route('/video_feed')
def video_feed():
    return Response(gesture_presentation.get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    gesture_presentation = GesturePresentation(folderP='Presentation',detection=True,doc_path="FIFA World Cup Analysis.pdf")
    app.run(debug=True)

# for normalmode
#     if folderp is empty and doc_path is not given then this will be activated
#     gesture_presentation = GesturePresentation(folderP='Presentation', detection=True)
