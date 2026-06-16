import os
from dotenv import load_dotenv
from core.graph import app

# Load API keys from .env
load_dotenv()

if __name__ == "__main__":
    test_file = "raw_patient_face.png"
    output_dir = "./clinical_runs/run_001"
    
    os.makedirs(output_dir, exist_ok=True)

    initial_payload = {
        "input_image_path": test_file,
        "output_directory": output_dir,
        "processed_image_paths": {},
        "face_landmarks": None,
        "gateway_error": None,
        "analysis_scores": {},
        "clinical_summary": None,
        "rubix_tx_hash": None,
        "agent_did": None
    }

    print(f"Starting Diagnostic Pipeline for: {test_file}")
    output_state = app.invoke(initial_payload)
    
    print("\n--- Pipeline Completed ---")
    print("Extracted Scores Object:", output_state.get("analysis_scores"))
    print("Written File Map Object:", output_state.get("processed_image_paths"))