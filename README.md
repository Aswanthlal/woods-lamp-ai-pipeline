# Multi-Spectral Facial Analysis Pipeline

An autonomous, multi-modal AI pipeline that simulates Wood's Lamp dermatological diagnostics. The system extracts seven quantitative facial metrics using localized computer vision techniques, generates an AI-powered dermatological assessment using a Large Language Model, and cryptographically anchors the resulting clinical report to a decentralized identity framework.

---

## Pipeline Architecture

### 1. Gateway Node (MediaPipe)
- Detects and validates facial landmarks.
- Maps structural facial regions for targeted analysis.

### 2. Vision Agents (OpenCV)
Executes seven specialized computer vision pipelines in parallel using:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Morphological transformations
- High-pass filtering
- Region-based facial segmentation

Each agent evaluates a specific skin metric and produces a score out of 10.

### 3. Consensus Node (Groq + Llama 3.3)
- Aggregates all computer vision outputs.
- Generates a comprehensive clinical-style dermatological assessment.
- Produces objective observations and recommendations.

### 4. Settlement Node (AgentDNA)
- Creates a cryptographic hash of the final report.
- Ensures report provenance, integrity, and immutability.

### 5. Reporter Node
- Compiles all metrics, visual outputs, and AI-generated findings.
- Produces a final Markdown diagnostic report.

---

## Tech Stack

| Component | Technology |
|------------|------------|
| Orchestration | LangGraph |
| Computer Vision | OpenCV, NumPy, MediaPipe |
| LLM Engine | Groq API (`llama-3.3-70b-versatile`) |
| Cryptographic Anchoring | AgentDNA |
| Programming Language | Python |

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/woods-lamp-diagnostic.git
cd cv_woods_lamp_mas
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Download the MediaPipe Model

Download the MediaPipe Face Landmarker model and place it in the project root directory:

```text
face_landmarker.task
```

---

##  Usage

Place the target facial image in the root directory, for example:

```text
raw_patient_face.png
```

Run the pipeline:

```bash
python main.py
```

---

## Output

The pipeline generates:

- Diagnostic overlay images for each analysis module
- Quantitative skin metric scores
- AI-generated dermatological assessment
- Cryptographic report hash
- Final Markdown clinical report


##  Extracted Facial Metrics

The system evaluates seven key dermatological indicators:

1. Pigmentation Analysis
2. Sebum Distribution
3. Pore Visibility
4. Skin Texture Uniformity
5. Erythema Detection
6. Hydration Estimation
7. Surface Damage Assessment

Each metric is scored on a standardized scale from **0–10**.

---

##  Data Integrity

Every generated report is cryptographically hashed through AgentDNA, providing:

- Report authenticity
- Tamper detection
- Immutable clinical provenance
- Decentralized verification support

---

# Sample Output Report

# Multi-Spectral Facial Analysis Report

**Date/Time:** 16-06-2026
**Patient ID:** Anon-001 (Demo)

---

## Quantitative Vision Metrics (0-10 Scale)
*Lower is generally better (less damage/severity).*

* **Erythema / Redness:** 10.00
* **Porphyrins / Bacteria:** 2.55
* **UV Damage / Spots:** 7.63
* **Brown Spots / Melanin:** 1.00
* **Spots & Blemishes:** 3.02
* **Texture / Roughness:** 1.00
* **Pore Size:** 1.82

---

## Attending AI Dermatologist Summary
**Clinical Diagnostic Report**

**Patient ID:** [Not Provided]
**Date:** [Not Provided]
**Clinical Dermatologist:** [Your Name]

**Introduction:**
This report presents a comprehensive analysis of the patient's facial skin metrics, compiled from advanced multi-spectral imaging and Wood's Lamp facial analytics. The quantitative data provides valuable insights into the patient's skin health, allowing for an objective assessment and diagnosis.

**Metrics Analysis:**

1. **UV Spots:** 7.63/10 - This moderate to high score indicates a significant presence of UV-induced skin damage, likely resulting from prolonged sun exposure. This may contribute to premature aging and increased risk of skin cancer.
2. **Erythema / Redness Index:** 10.00/10 - The maximum score suggests severe erythema, which may be indicative of inflammation, rosacea, or other skin conditions. This requires further investigation to determine the underlying cause.
3. **Brown Spots / Melanin Deposition:** 1.00/10 - The low score indicates minimal melanin deposition, suggesting a low risk of hyperpigmentation or melasma.
4. **Porphyrins / Bacterial Activity:** 2.55/10 - This moderate score may indicate some level of bacterial activity, potentially contributing to acne or other skin infections.
5. **Spots and Blemishes:** 3.02/10 - The relatively low score suggests a moderate presence of spots and blemishes, which may be related to acne, blackheads, or other skin imperfections.
6. **Texture Roughness Profile:** 1.00/10 - The low score indicates a smooth skin texture, suggesting minimal signs of aging or environmental damage.
7. **Pore Size Metric:** 1.82/10 - The relatively low score suggests small to moderate pore size, which is generally considered a positive indicator of skin health.

**Clinical Impression:**
Based on the analysis of the multi-spectral facial metrics, the patient's skin presents with:

* Significant UV-induced skin damage and erythema, which may be contributing to inflammation and premature aging.
* Minimal melanin deposition and hyperpigmentation.
* Moderate bacterial activity, which may be related to acne or other skin infections.
* A relatively smooth skin texture and small to moderate pore size.

**Recommendations:**

1. **Sun protection:** Emphasize the importance of daily sun protection, using a broad-spectrum sunscreen with at least SPF 30, to prevent further UV-induced damage.
2. **Erythema management:** Investigate the underlying cause of the severe erythema and develop a treatment plan to address inflammation and reduce redness.
3. **Bacterial control:** Consider topical or oral antibiotics to control bacterial activity and prevent acne or other skin infections.
4. **Skin care routine:** Establish a gentle, non-comedogenic skin care routine to maintain skin health and prevent further damage.

**Follow-up:**
A follow-up appointment is recommended to monitor the patient's response to treatment and adjust the plan as necessary.

**Signature:**
[Your Name]
Clinical Dermatologist

---

## Cryptographic Settlement (AgentDNA)
* **Agent DID:** `bafybmibqakpx6sl4zcl25gst77s5tpywgqkfp42p4gf3tmxz7cb4irpqqzi`
* **Report Hash (SHA-256):** `dd14b1c95e7f2435e5b498fea0557563eaf99096b98e364ecad7697e06103c29`
* **Status:** Anchored & Verified
