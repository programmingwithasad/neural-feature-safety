from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.gateway.safety_gateway import SafetyGateway


app = FastAPI(
    title="Neural Feature Safety API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://192.168.1.34:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

gateway = SafetyGateway()


class TextRequest(BaseModel):
    text: str


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {
        "name": "Neural Feature Safety API",
        "status": "online",
        "version": "1.0.0",
        "gateway": "active"
    }


@app.get("/health")
def health():
    safety_model = gateway.input_guard.safety_model

    return {
        "status": "healthy",
        "gateway": "active",
        "device": str(safety_model.device),
        "threshold": safety_model.threshold,
        "llm": "Llama-3.2-3B-Instruct",
        "safety_classifier": "production"
    }


@app.post("/analyze")
def analyze(request: TextRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    return gateway.input_guard.analyze(text)

@app.post("/v1/explain")
def explain(request: TextRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    return gateway.input_guard.safety_model.explain(text)
@app.post("/v1/explain/generate")
def generate_explanation(request: TextRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    try:
        return gateway.explain_with_llama(text)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )
    
@app.post("/v1/analyze")
def v1_analyze(request: TextRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    return gateway.input_guard.analyze(text)


@app.post("/v1/chat")
def chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    try:
        return gateway.chat(message)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )


@app.get("/v1/health")
def v1_health():
    safety_model = gateway.input_guard.safety_model

    return {
        "status": "healthy",
        "gateway": "active",
        "device": str(safety_model.device),
        "threshold": safety_model.threshold,
        "llm": {
            "name": "Llama-3.2-3B-Instruct",
            "mode": "local",
            "adapter": "llama_safety_adapter_v2"
        },
        "safety": {
            "classifier": "final_safety_classifier.pkl",
            "input_guard": "active",
            "output_guard": "active"
        }
    }