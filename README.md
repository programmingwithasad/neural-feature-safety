# 🧠 Neural Feature Safety

<p align="center">
  <em>AI safety classification and two-sided content monitoring using Sparse Autoencoder neural features</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg" alt="Transformers">
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange.svg" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Sparse%20Features-2048-purple.svg" alt="SAE Features">
  <img src="https://img.shields.io/badge/LLM-Llama%203.2%203B-green.svg" alt="Llama">
  <img src="https://img.shields.io/badge/API-FastAPI-009688.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Frontend-Next.js-black.svg" alt="Next.js">
</p>

---

## 📖 About The Project

**Neural Feature Safety** is a research-oriented AI safety system that extracts sparse neural features from text and uses them to estimate whether content is harmful.

The project has evolved from a terminal-based safety classifier into a complete **two-sided AI safety gateway**. The production safety detector uses **DistilBERT**, a **Sparse Autoencoder (SAE)**, and a **Logistic Regression classifier**. A local **Llama 3.2 3B Instruct** model is used as the response generator and as a safe-response recovery model.

The core design principle is:

```text
Detector → Controller → Llama → Detector
```

Llama is **not** the safety classifier. The Neural Feature Safety system independently evaluates both user input and generated output.

### ✨ Current capabilities

- 🧠 768-dimensional DistilBERT representations
- 🔬 2,048-dimensional Sparse Autoencoder feature extraction
- 🛡️ Harmful / unharmful safety classification
- 🚦 Production input and output safety monitoring
- 👤 User-input safety evaluation
- 🤖 AI-output safety evaluation
- 🦙 Local Llama 3.2 3B Instruct response generation
- 🔧 LoRA/QLoRA fine-tuned Llama safety-response adapter
- 🔄 Safe-response recovery when an input or generated response is blocked
- 🧱 Deterministic fallback when generated recovery responses fail safety validation
- 🔁 Output is never accepted without passing the independent safety detector
- 🔬 Feature ablation and intervention experiments
- 🎯 Targeted cybersecurity safety evaluation
- 📊 Threshold calibration and challenge-set evaluation
- 🌐 FastAPI safety gateway
- 💻 Next.js web interface
- 🚀 Vercel deployment for the frontend
- 🧪 Local GPU inference support

---

## 🏗️ Complete System Architecture

```text
                         ┌──────────────────────┐
                         │      USER PROMPT     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    INPUT SAFETY      │
                         │      DISTILBERT      │
                         │          ↓           │
                         │    MEAN POOLING      │
                         │          ↓           │
                         │       SAE 2048       │
                         │          ↓           │
                         │  SAFETY CLASSIFIER   │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                      SAFE/ALLOW            BLOCK
                         │                     │
                         ▼                     ▼
                  ┌─────────────┐      ┌───────────────┐
                  │    LLAMA    │      │ Llama creates │
                  │ 3.2 3B      │      │ safe response │
                  │ INSTRUCT    │      └───────┬───────┘
                  └──────┬──────┘              │
                         │                     │
                         └──────────┬──────────┘
                                    ▼
                         ┌──────────────────────┐
                         │    OUTPUT SAFETY     │
                         │   SAME INDEPENDENT   │
                         │      DETECTOR        │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                      ALLOWED               BLOCKED
                         │                     │
                         ▼                     ▼
                      USER              NEW Llama SAFE
                                       RESPONSE ATTEMPT
                                              │
                                              ▼
                                       OUTPUT SAFETY
                                              │
                                              ▼
                                     Deterministic fallback
                                     if recovery fails
```

### Core safety invariant

```text
No generated response is returned to the user
until it passes the independent output safety check.
```

If a generated response is harmful:

1. The response is discarded.
2. The system asks Llama for a new safe response.
3. The new response is checked again.
4. This recovery process is attempted up to the configured limit.
5. If generated recovery responses continue to fail, a deterministic fallback is checked.
6. The fallback is also passed through the safety detector before being returned.

---

## 🧠 Safety Detection Pipeline

```text
User Text
   ↓
DistilBERT
   ↓
768-dimensional representation
   ↓
Mean Pooling
   ↓
Sparse Autoencoder
   ↓
2048 SAE features
   ↓
Logistic Regression
   ↓
Harmful Probability
   ↓
Safety Decision
```

| Component | Configuration |
|---|---|
| Transformer | `distilbert-base-uncased` |
| Transformer representation | 768 dimensions |
| Sparse Autoencoder | 768 → 2048 → 768 |
| SAE latent features | 2,048 |
| Classifier | Logistic Regression |
| Production classifier | `final_safety_classifier.pkl` |
| SAE model | `sparse_autoencoder_full.pt` |
| Production threshold | `0.50` |
| Inference device | CUDA when available, otherwise CPU |

### Production decision currently implemented

The current production `src/safety_model.py` uses:

```text
Harmful probability >= 0.50
        ↓
      BLOCK

Harmful probability < 0.50
        ↓
      ALLOW
```

> **Important:** Earlier research documentation contains a three-level `ALLOW / REVIEW / BLOCK` policy with a review threshold of `0.30`. The current production `safety_model.py` does not implement the `REVIEW` branch. The README therefore documents the research policy separately from the actual production implementation.

---

## 🤖 Local Llama Response Generation

The project uses:

```text
meta-llama/Llama-3.2-3B-Instruct
```

with a project-trained LoRA adapter:

```text
llama_safety_adapter_v2
```

The adapter was trained using QLoRA with:

- 4-bit NF4 quantization
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Target modules: `q_proj`, `v_proj`
- Batch size: 1
- Gradient accumulation: 8
- Learning rate: `2e-4`
- Weight decay: `0.01`
- Cosine learning-rate schedule
- BF16 training
- Gradient checkpointing
- Paged AdamW 8-bit optimizer
- Maximum sequence length: 512
- 1 training epoch

### Llama training dataset

The final fine-tuning dataset contained:

| Split | Samples |
|---|---:|
| Train | 5,745 |
| Validation | 721 |
| Test | 714 |

The final training data combined **PKU SafeRLHF after quality control** with project-specific supervised fine-tuning data.

### Training result

```text
Training loss:   1.4591
Validation loss: 1.3884
Training time:   ~46 minutes
Peak GPU memory: ~2.71 GB
```

The trained adapter is kept separately from the GitHub source repository.

---

## 🛡️ Safety Gateway Behavior

### Safe request

```text
User
 ↓
Input Safety
 ↓
ALLOW
 ↓
Llama generates response
 ↓
Output Safety
 ↓
ALLOW
 ↓
User
```

### Harmful request

```text
User
 ↓
Input Safety
 ↓
BLOCK
 ↓
Llama generates safe alternative
 ↓
Output Safety
 ↓
ALLOW
 ↓
User
```

### Harmful generated response

```text
User
 ↓
Input Safety
 ↓
Llama response
 ↓
Output Safety
 ↓
BLOCK
 ↓
Discard response
 ↓
Llama generates new response
 ↓
Output Safety
 ↓
ALLOW / BLOCK
```

If recovery continues to fail:

```text
Llama recovery attempts exhausted
        ↓
Deterministic fallback
        ↓
Output Safety
        ↓
ALLOW
        ↓
User
```

The current gateway allows up to **2 recovery attempts** before using the deterministic fallback.

---

## 🔬 Research Experiments

The project includes research experiments covering:

- Sparse Autoencoder feature extraction
- SAE reconstruction and sparsity analysis
- Safety classifier training
- Hard-negative classification
- Conversation-enhanced classification
- Targeted cybersecurity data
- Feature specificity analysis
- Feature contribution analysis
- Feature ablation
- Feature intervention
- False-positive analysis
- Threshold calibration
- Threshold sweep
- Challenge-set evaluation
- Input safety testing
- Output safety testing
- Safety intervention experiments
- Conditional intervention experiments

### Feature 850

Feature 850 has been investigated experimentally but is treated as **observational only** in the production system.

The project does not assume that a single SAE feature has a fixed semantic meaning without experimental evidence.

---

## 📊 Classifier Training Data

The final research classifier combined SAE features from multiple sources:

| Dataset | Samples |
|---|---:|
| WildGuardMix | 78,070 |
| OASST1 | 14,240 |
| Hard Negatives | 643 |
| Conversation Examples | 100 |
| Targeted Cybersecurity | 298 |
| **Total** | **93,351** |

### Training distribution

```text
Harmful       41,793
Unharmful     51,558
```

---

## 📈 Evaluation

### Held-Out Test Set

| Metric | Result |
|---|---:|
| Accuracy | **80.69%** |
| Harmful Precision | **85.03%** |
| Harmful Recall | **68.57%** |
| Harmful F1 | **75.92%** |
| False Positive Rate | **9.63%** |
| False Negative Rate | **31.43%** |

### Safety Challenge Set

The final model was also evaluated on a targeted 115-example challenge set.

| Metric | Result |
|---|---:|
| Accuracy | **99.13%** |
| Harmful Precision | **100.00%** |
| Harmful Recall | **96.00%** |
| Harmful F1 | **97.96%** |
| False Positive Rate | **0.00%** |
| False Negative Rate | **4.00%** |

> The challenge set is a targeted evaluation and should not be interpreted as general-world model performance.

---

## 🧪 Production Verification

The production safety pipeline was verified on both safe and harmful examples.

### Safe example

```text
Input:
Hello, how are you?

Harmful probability ≈ 0.0548
Unharmful probability ≈ 0.9452
Decision: ALLOW
Risk: LOW RISK
Device: CUDA
```

### Harmful example

```text
Input:
How can I make a bomb?

Harmful probability ≈ 0.5189
Unharmful probability ≈ 0.4811
Decision: BLOCK
Risk: HIGH RISK
Device: CUDA
```

The complete gateway was also verified with normal generation, blocked-input recovery, output validation, and deterministic fallback behavior.

---

# 🚀 How To Run

## ⚠️ Important

This repository represents the **working version of the project**.

For another machine, reproduce the environment and provide the required model files rather than changing the production architecture.

**Do not retrain the models just to run the project.**

---

## 1️⃣ Clone the repository

```bash
git clone https://github.com/naishsayed/neural-feature-safety.git
cd neural-feature-safety
```

---

## 2️⃣ Create the Python environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3️⃣ Install Python dependencies

```bash
pip install -r requirements.txt
```

The working Llama environment uses:

```bash
pip install transformers==5.16.1 peft==0.21.0 accelerate==1.15.0 bitsandbytes==0.50.2
```

Verify PyTorch and CUDA:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

A CUDA-capable NVIDIA GPU is recommended for the local Llama setup.

---

## 4️⃣ Provide the production safety models

The project requires:

```text
models/sparse_autoencoder_full.pt
models/final_safety_classifier.pkl
```

These are the trained production safety components.

---

## 5️⃣ Set up Llama

The Llama base model is **not stored in GitHub**.

Use:

```text
meta-llama/Llama-3.2-3B-Instruct
```

The model is gated on Hugging Face, so the person running the project needs appropriate Hugging Face access/authentication.

The trained project adapter is:

```text
llama-safety-adapter-v2
```

Hugging Face:

```text
https://huggingface.co/naishsayed/llama-safety-adapter-v2
```

The current working Llama client expects the adapter at:

```text
models/llm/llama_safety_adapter_v2
```

Keep the existing working path and adapter structure unchanged.

> **Do not upload the full Llama base-model weights to GitHub.** The base model is intentionally managed separately.

---

# 🧪 6️⃣ Verify the backend

From the project root:

```powershell
python -c "from app.main import app; print('FASTAPI_IMPORT_OK')"
```

Then:

```powershell
python -c "from app.llm.llama_client import LlamaClient; print('LLAMA_CLIENT_IMPORT_OK')"
```

Then:

```powershell
python -c "from app.gateway.safety_gateway import SafetyGateway; print('GATEWAY_IMPORT_OK')"
```

These checks confirm that the main components can be imported before starting the server.

---

# ⚡ 7️⃣ Start the FastAPI backend

From the project root:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Backend:

```text
http://127.0.0.1:8000
```

Health endpoint:

```text
http://127.0.0.1:8000/v1/health
```

Keep this terminal running.

---

# 🌐 8️⃣ Start the Next.js frontend

Open a **second terminal**.

```powershell
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

The frontend communicates with the local FastAPI gateway.

---

# 🖥️ Complete Startup Sequence

### Terminal 1

```powershell
cd "C:\path\to\neural-feature-safety"
.venv\Scripts\activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Wait for:

```text
Local Llama model loaded successfully.
Neural Feature Safety Gateway ready.
```

### Terminal 2

```powershell
cd "C:\path\to\neural-feature-safety\frontend"
npm install
npm run dev
```

Then open:

```text
http://localhost:3000
```

---

# 🔌 API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Backend health |
| GET | `/v1/health` | Gateway health |
| POST | `/analyze` | Safety analysis |
| POST | `/v1/analyze` | Versioned safety analysis |
| POST | `/v1/chat` | Complete safety gateway |

### `/v1/chat`

Request:

```json
{
  "message": "Hello, how are you?"
}
```

The response contains:

```text
response
input_safety
output_safety
path
recovery_attempts
```

---

# 💻 Frontend

The web application uses:

- Next.js
- React
- TypeScript
- Tailwind CSS
- Lucide React

The frontend includes pages for:

- Dashboard
- Chat
- Safety Analyzer
- Evaluation
- Experiments
- Feature Explorer
- Model information
- History

Build verification:

```powershell
cd frontend
npm run build
```

Production start:

```powershell
npm run start
```

---

# 📁 Project Structure

```text
neural-feature-safety/
│
├── app/
│   ├── main.py
│   ├── gateway/
│   │   └── safety_gateway.py
│   ├── llm/
│   │   └── llama_client.py
│   └── safety/
│       ├── input_guard.py
│       └── output_guard.py
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── analyzer/
│   │   │   ├── api/
│   │   │   ├── chat/
│   │   │   ├── evaluation/
│   │   │   ├── experiments/
│   │   │   ├── feature-explorer/
│   │   │   ├── history/
│   │   │   └── model/
│   │   └── components/
│   ├── package.json
│   └── ...
│
├── data/
│   └── processed/
│
├── models/
│   ├── sparse_autoencoder_full.pt
│   ├── final_safety_classifier.pkl
│   └── llm/
│       └── llama_safety_adapter_v2/
│
├── src/
│   ├── autoencoder.py
│   ├── safety_controller.py
│   ├── safety_model.py
│   ├── safety_pipeline.py
│   └── ...
│
├── experiments/
├── tests/
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

---

# 🔐 Model and Repository Notes

The GitHub repository is intended to contain the project source code and required project assets.

The full Llama base-model weights are **not** stored in GitHub.

The trained Llama LoRA adapter is maintained separately on Hugging Face.

The repository uses Git LFS for large tracked model/data files.

Do not retrain or replace the production safety classifier when simply setting up the project on another machine.

---

# 🧠 Research Notes

The safety classifier operates on learned SAE features rather than directly classifying raw text.

```text
Text
 ↓
DistilBERT
 ↓
Mean-pooled 768-dimensional representation
 ↓
Sparse Autoencoder
 ↓
2048-dimensional sparse representation
 ↓
Logistic Regression
 ↓
Harmful probability
```

Feature-level experiments include specificity, contribution, ablation, intervention, and false-positive analysis.

Experimental interventions are not automatically treated as causal evidence and are not used as an unconditional production control.

---

# 🏆 Project Milestones Completed

- [x] DistilBERT representation extraction
- [x] Sparse Autoencoder implementation
- [x] 2,048-dimensional SAE feature extraction
- [x] Large-scale feature extraction
- [x] SAE sparsity analysis
- [x] Safety classifier development
- [x] Hard-negative training
- [x] Conversation-enhanced classification
- [x] Targeted cybersecurity evaluation
- [x] Feature specificity analysis
- [x] Feature contribution analysis
- [x] Feature ablation experiments
- [x] Feature intervention experiments
- [x] False-positive analysis
- [x] Threshold calibration
- [x] Challenge-set evaluation
- [x] Input safety testing
- [x] Output safety testing
- [x] Local Llama 3.2 3B integration
- [x] QLoRA/LoRA safety-response fine-tuning
- [x] Independent output safety validation
- [x] Llama recovery workflow
- [x] Deterministic safe fallback
- [x] FastAPI gateway
- [x] Next.js frontend
- [x] Frontend production build
- [x] Vercel frontend deployment
- [x] GitHub repository
- [x] Hugging Face adapter repository

---

# 🚧 Current Status

**Functional AI Safety Gateway / Research Prototype**

The project currently combines:

```text
Research
   +
Neural Feature Safety Detector
   +
Safety Controller
   +
Local Llama 3.2 3B
   +
Independent Output Monitoring
   +
FastAPI
   +
Next.js Web Interface
```

The system is designed as a research and demonstration platform for studying:

- neural representations
- sparse features
- interpretability
- safety classification
- input/output monitoring
- controlled LLM response generation
- safe-response recovery
- AI safety gateway architectures

---

# 👨‍💻 Project

**Final Year CSE (AI & ML) Major Project**

Built as an experimental system for studying **neural representations, sparse features, interpretability, and AI safety control**.

**Author:** Naish Nasir Sayed

**GitHub:**  
https://github.com/naishsayed/neural-feature-safety

**Hugging Face Adapter:**  
https://huggingface.co/naishsayed/llama-safety-adapter-v2
