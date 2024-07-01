import  os
import cv2
from thickness import *
folderPath = "sample"
pathImages = sorted(os.listdir(folderPath), key=len)
print(len(pathImages)==0)
width, height = 1280, 720  # Assuming imgCurrent.shape is (1080, 1920, 3)
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)
def Show():
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        if not success:
            print("Failed to capture image from webcam")
            break

        cv2.imshow("Output Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        return  img




# pathFullImage = os.path.join(folderPath, pathImages[0])
# imgCurrent = cv2.imread(pathFullImage)
# height2, width2 = imgCurrent.shape[:2]
# new_width = width2 // 2
# new_height = height2 // 2
# imgCurrent = cv2.resize(imgCurrent, (new_width, new_height))
# controller = GestureThicknessController()
# while True:
#     ret, frame = controller.cap.read()
#     if not ret:
#         break
#     # frame=cv2.flip(frame,1)
#     output_image = imgCurrent.copy()
#     output_image, thickness = controller.process_frame(frame, "brush", output_image)
#     # print(int(thickness))
#     cv2.imshow("Output Image", output_image)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# controller.release()

