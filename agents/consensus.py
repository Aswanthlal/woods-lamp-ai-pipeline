import os
from openai import OpenAI
from typing import Dict, Any
from core.state import DiagnosticState

def consensus_node(state: DiagnosticState) -> Dict[str, Any]:
    print("\n--- [Consensus Node] Ingesting Wood's Lamp CV Scores into API Engine ---")
    scores = state["analysis_scores"]
    if not scores:
        return {"clinical_summary": "Error: No analytical vision metrics available to summarize."}

    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )

    system_prompt = (
        "You are an expert Clinical Dermatologist specializing in advanced multi-spectral "
        "and Wood's Lamp facial analytics. Your job is to synthesize quantitative metrics "
        "into a structured, professional, objective clinical diagnostic report."
    )

    user_content = f"""
    Please analyze the following multi-spectral facial metrics compiled from the vision pipeline:
    - UV Spots: {scores.get('uv_spots', 0.0):.2f}/10
    - Erythema / Redness Index: {scores.get('redness', 0.0):.2f}/10
    - Brown Spots / Melanin Deposition: {scores.get('brown_spots', 0.0):.2f}/10
    - Porphyrins / Bacterial Activity: {scores.get('porphyrins', 0.0):.2f}/10
    - Spots and Blemishes: {scores.get('spots_and_blemishes', 0.0):.2f}/10
    - Texture Roughness Profile: {scores.get('texture', 0.0):.2f}/10
    - Pore Size Metric: {scores.get('pore_size', 0.0):.2f}/10
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2
        )
        return {"clinical_summary": response.choices[0].message.content}
    except Exception as e:
        return {"clinical_summary": f"Failed to generate report: {str(e)}"}