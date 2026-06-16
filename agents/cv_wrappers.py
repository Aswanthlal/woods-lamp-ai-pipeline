import os
import cv2
from typing import Dict, Any
from core.state import DiagnosticState

def create_cv_agent(metric_key: str, engine_function):
    def cv_agent_node(state: DiagnosticState) -> Dict[str, Any]:
        print(f"\n[Agent] {metric_key.replace('_', ' ').title()} Specialist executing...")

        img_path = state["input_image_path"]
        landmarks = state["face_landmarks"]
        out_dir = state["output_directory"]

        source_matrix = cv2.imread(img_path)
        score, diagnostic_overlay = engine_function(source_matrix, landmarks)

        target_path = os.path.join(out_dir, f"{metric_key}_diagnostic.jpg")
        cv2.imwrite(target_path, diagnostic_overlay)

        return {
            "analysis_scores": {metric_key: score},
            "processed_image_paths": {metric_key: target_path}
        }
    return cv_agent_node