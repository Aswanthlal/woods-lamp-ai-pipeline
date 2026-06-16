from typing import Dict, Any, Optional, Annotated
from typing_extensions import TypedDict

def merge_dicts(dict1: dict, dict2: dict) -> dict:
    if not dict1: return dict2
    if not dict2: return dict1
    merged = dict1.copy()
    merged.update(dict2)
    return merged

class DiagnosticState(TypedDict):
    input_image_path: str
    output_directory: str
    face_landmarks: Optional[list]
    gateway_error: Optional[str]

    # Annotated allows parallel agents to write simultaneously
    analysis_scores: Annotated[Dict[str, float], merge_dicts]
    processed_image_paths: Annotated[Dict[str, str], merge_dicts]

    clinical_summary: Optional[str]
    rubix_tx_hash: Optional[str]
    agent_did: Optional[str]
    report_file_path: Optional[str]