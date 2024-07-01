import cv2
import time
import numpy as np
import math
import HandTrackingModule as htp
import mediapipe as mp

class GestureThicknessController:
    def __init__(self, Wcam=1280, Hcam=720, detection_confidence=0.7):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, Wcam)
        self.cap.set(4, Hcam)
        self.detector = htp.handDectector(detectionconfidence=detection_confidence)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()
        self.initial_nose_y = None
        self.pTime = 0
        self.thickness = {
            "brush": 0,
            "eraser": 0
        }
        self.thickness_percent = 0
        self.active = True

    # Function to calculate the Euclidean distance
    @staticmethod
    def euclidean_dist(ptA, ptB):
        return ((ptA[0] - ptB[0]) ** 2 + (ptA[1] - ptB[1]) ** 2) ** 0.5

    def process_frame(self, frame, thickness_type, output_image):
        if not self.active:
            return output_image, self.thickness[thickness_type], False  # Return False if not active

        # frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and find face landmarks
        result = self.face_mesh.process(rgb_frame)

        nod_detected = False  # Flag to track nod detection

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                h, w, _ = frame.shape
                # Convert landmarks to numpy array for easier processing
                landmarks = np.array([(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark])

                # Extract the coordinates of left eye, right eye, and nose tip
                left_eye = landmarks[33]  # Left eye
                right_eye = landmarks[263]  # Right eye
                nose = landmarks[1]  # Nose tip

                # Initialize the initial nose position
                if self.initial_nose_y is None:
                    self.initial_nose_y = nose[1]

                # Calculate the vertical movement of the nose
                nod = nose[1] - self.initial_nose_y
                nod_abs = abs(nod)

                # Check for vertical movement (head nod)
                if nod_abs > 20 and nod > 0:
                    self.thickness[thickness_type] = int(self.thickness_percent)
                    self.active = False  # Stop processing after head nod detection
                    nod_detected = True  # Set nod detected flag

                # Draw the facial landmarks on the output image
                for pt in landmarks:
                    cv2.circle(output_image, tuple(pt), 1, (0, 255, 0), -1)

        # Hand tracking and volume control part
        self.detector.findHands(frame)
        lmList = self.detector.findPosition(frame, draw=False)

        if lmList:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            c1, c2 = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(output_image, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(output_image, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(output_image, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(output_image, (c1, c2), 15, (255, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)
            if length < 50:
                cv2.circle(output_image, (c1, c2), 15, (0, 255, 0), cv2.FILLED)

            brushThicknessBar = np.interp(length, [50, 168], [400, 150])
            self.thickness_percent = np.interp(length, [50, 168], [0, 100])
            self.thickness[thickness_type] = np.interp(length, [50, 168], [0, 100])
            cv2.rectangle(output_image, (100, int(brushThicknessBar)), (125, 400), (0, 255, 0), cv2.FILLED)

        cv2.rectangle(output_image, (100, 150), (125, 400), (0, 255, 0), 3)
        cv2.putText(output_image, f"{int(self.thickness_percent)} ", (90, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Display FPS
        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime
        cv2.putText(output_image, f"{thickness_type.capitalize()} Thickness set to: {int(self.thickness[thickness_type])}"
                    , (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        return output_image, self.thickness[thickness_type], not nod_detected  # Return True if no nod detected, False otherwise

    def reset(self):
        self.active = True
        self.initial_nose_y = None

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.active = False


# Example usage
if __name__ == "__main__":
    controller = GestureThicknessController()

    # Load your custom image
    custom_image = cv2.imread('Presentation/1.png')

    while True:
        ret, frame = controller.cap.read()
        if not ret:
            break

        output_image = custom_image.copy()
        output_image = cv2.resize(output_image, (960, 540))
        output_image, thickness, continue_processing = controller.process_frame(frame, "brush", output_image)
        print(int(continue_processing))
        cv2.imshow("Output Image", output_image)

        if cv2.waitKey(1) & 0xFF == ord('q') or not continue_processing:
            break

    controller.release()
