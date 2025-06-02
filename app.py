"""
FastAPI application for the digital twin API.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import torch
from pathlib import Path

from ..model.trainer import DigitalTwinModel, DigitalTwinTrainer

app = FastAPI(
    title="CodeBrain Digital Twin API",
    description="API for interacting with your coding digital twin",
    version="1.0.0"
)

# Load model and trainer
model = None
trainer = None

class CodeRequest(BaseModel):
    """Request model for code generation."""
    prompt: str
    style_constraints: Optional[Dict] = None
    pattern_constraints: Optional[Dict] = None
    max_length: Optional[int] = 512

class CodeResponse(BaseModel):
    """Response model for generated code."""
    code: str
    style_metrics: Dict
    pattern_metrics: Dict
    confidence: float

class TrainingRequest(BaseModel):
    """Request model for model training."""
    data_path: str
    config: Dict

class TrainingResponse(BaseModel):
    """Response model for training status."""
    status: str
    losses: List[float]
    message: str

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global model, trainer
    try:
        model_path = Path("models/digital_twin.pt")
        if model_path.exists():
            model = DigitalTwinModel(
                base_model_name="gpt2",  # Replace with your base model
                style_dim=10,  # Replace with your style dimension
                pattern_dim=10  # Replace with your pattern dimension
            )
            trainer = DigitalTwinTrainer(model)
            trainer.load_model(str(model_path))
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/generate", response_model=CodeResponse)
async def generate_code(request: CodeRequest):
    """Generate code based on the prompt and constraints."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        # Prepare inputs
        inputs = model.tokenizer(
            request.prompt,
            max_length=request.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert constraints to tensors
        style_tensor = torch.tensor(
            list(request.style_constraints.values()) if request.style_constraints else [0] * 10,
            dtype=torch.float
        )
        pattern_tensor = torch.tensor(
            list(request.pattern_constraints.values()) if request.pattern_constraints else [0] * 10,
            dtype=torch.float
        )
        
        # Generate code
        with torch.no_grad():
            outputs = model(
                inputs['input_ids'],
                inputs['attention_mask'],
                style_tensor.unsqueeze(0),
                pattern_tensor.unsqueeze(0)
            )
            
        # Decode output
        generated_code = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate metrics
        style_metrics = model.style_encoder(style_tensor.unsqueeze(0)).squeeze().tolist()
        pattern_metrics = model.pattern_encoder(pattern_tensor.unsqueeze(0)).squeeze().tolist()
        
        # Calculate confidence (placeholder)
        confidence = 0.8
        
        return CodeResponse(
            code=generated_code,
            style_metrics=dict(zip(request.style_constraints.keys(), style_metrics)) if request.style_constraints else {},
            pattern_metrics=dict(zip(request.pattern_constraints.keys(), pattern_metrics)) if request.pattern_constraints else {},
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """Train the digital twin model."""
    try:
        # Initialize model if not exists
        global model, trainer
        if model is None:
            model = DigitalTwinModel(
                base_model_name=request.config['base_model'],
                style_dim=request.config['style_dim'],
                pattern_dim=request.config['pattern_dim']
            )
            trainer = DigitalTwinTrainer(model, learning_rate=request.config['learning_rate'])
            
        # Train model
        losses = trainer.train(
            train_loader=request.config['train_loader'],
            num_epochs=request.config['num_epochs'],
            device=request.config['device']
        )
        
        # Save model
        trainer.save_model("models/digital_twin.pt")
        
        return TrainingResponse(
            status="success",
            losses=losses,
            message="Model trained successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 