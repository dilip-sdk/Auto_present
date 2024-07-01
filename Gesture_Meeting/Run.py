from main import GesturePresentation
import  cv2
gesture_presentation = GesturePresentation(detection=True, doc_path="FIFA World Cup Analysis.pdf")
# Run the presentation
result_image = gesture_presentation.run()

# Save or process the result image as needed
cv2.imwrite("result_image.jpg", result_image)
