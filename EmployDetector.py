
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGINX = 10
MARGINY = 10
ROW_SIZE = 10  
FONT_SIZE = 1
FONT_THICKNESS = 2
TEXT_COLOR = (0, 255, 0) 


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGINX + bbox.origin_x,
                     MARGINY + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_COMPLEX,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
while cap.isOpened():
    ret, frame = cap.read()
    base_options = python.BaseOptions(model_asset_path='modelHumans2.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                        score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    detection_result = detector.detect(image)

    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    cv2.imshow('rest', rgb_annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
cap.release()
cv2.destryAllWindows()

