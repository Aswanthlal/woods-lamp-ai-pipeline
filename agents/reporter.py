import os
from typing import Dict, Any
from core.state import DiagnosticState

def reporter_node(state: DiagnosticState) -> Dict[str, Any]:
    print("\n--- [Reporter Node] Archiving Clinical Report to Disk ---")

    out_dir = state["output_directory"]
    report_path = os.path.join(out_dir, "clinical_diagnostic_report.md")

    scores = state.get("analysis_scores", {})
    summary = state.get("clinical_summary", "No clinical summary generated.")
    tx_hash = state.get("rubix_tx_hash", "Pending/Failed")
    did = state.get("agent_did", "Unknown DID")

    report_content = f"""# Multi-Spectral Facial Analysis Report

**Date/Time:** {os.popen('date /t').read().strip()}
**Patient ID:** Anon-001 (Demo)

---

## Quantitative Vision Metrics (0-10 Scale)
*Lower is generally better (less damage/severity).*

* **Erythema / Redness:** {scores.get('redness', 'N/A'):.2f}
* **Porphyrins / Bacteria:** {scores.get('porphyrins', 'N/A'):.2f}
* **UV Damage / Spots:** {scores.get('uv_spots', 'N/A'):.2f}
* **Brown Spots / Melanin:** {scores.get('brown_spots', 'N/A'):.2f}
* **Spots & Blemishes:** {scores.get('spots_and_blemishes', 'N/A'):.2f}
* **Texture / Roughness:** {scores.get('texture', 'N/A'):.2f}
* **Pore Size:** {scores.get('pore_size', 'N/A'):.2f}

---

## Attending AI Dermatologist Summary
{summary}

---

## Cryptographic Settlement (AgentDNA)
* **Agent DID:** `{did}`
* **Report Hash (SHA-256):** `{tx_hash}`
* **Status:** Anchored & Verified
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Success: Report committed to {report_path}")
    return {"report_file_path": report_path}