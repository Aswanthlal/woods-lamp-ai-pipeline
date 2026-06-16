import cv2
import mediapipe as mp
from typing import Dict, Any
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from core.state import DiagnosticState

def gateway_node(state: DiagnosticState) -> Dict[str, Any]:
    print("\n[Gateway Node] Validating Target Image Matrix")
    img_path = state["input_image_path"]

    image = cv2.imread(img_path)
    if image is None:
        return {"gateway_error": f"IO Error: Unable to read file matrix at {img_path}"}

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)

    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            results = landmarker.detect(mp_image)

        if not results.face_landmarks:
            return {"gateway_error": "Validation Error: Face structural landmarks missing."}

        extracted_landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in results.face_landmarks[0]]
        print("Success: Structural face topology validated and locked.")
        return {"face_landmarks": extracted_landmarks, "gateway_error": None}
    except Exception as e:
        return {"gateway_error": f"MediaPipe Execution Error: {str(e)}"}