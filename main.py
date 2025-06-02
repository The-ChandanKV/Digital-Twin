from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import json
from codebrain.model.trainer import train_digital_twin
from typing import Optional, List, Dict
import ast
import autopep8
import black
import mimetypes
import chardet
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Digital Twin API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TRAIN_STATUS = {"status": "idle", "losses": [], "error": None}

class ConfigModel(BaseModel):
    base_model: str
    style_dim: int
    pattern_dim: int
    max_length: int
    batch_size: int
    num_workers: int
    learning_rate: float
    num_epochs: int
    device: str

class CodeCorrectionRequest(BaseModel):
    code: str
    language: str = "python"
    auto_apply: bool = False

class CodeCorrectionResponse(BaseModel):
    original_code: str
    corrected_code: str
    suggestions: List[Dict[str, str]]
    applied: bool

def detect_file_encoding(file_path: str) -> str:
    """Detect the encoding of a file."""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding'] or 'utf-8'

def validate_and_repair_json_file(file_path: str, file_type: str) -> bool:
    """Validate if a file is a valid JSON file. If not, delete it and return False."""
    try:
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} does not exist.")
            return False
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                logger.error(f"File {file_path} is empty. Deleting.")
                os.remove(file_path)
                return False
            json.loads(content)
            return True
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {str(e)}. Deleting.")
        os.remove(file_path)
        return False
    except Exception as e:
        logger.error(f"Error reading {file_path}: {str(e)}. Deleting.")
        os.remove(file_path)
        return False

def convert_file_to_training_format(file_path: str, file_type: str) -> Dict:
    """Convert any file to the training format."""
    try:
        logger.info(f"Converting file {file_path} of type {file_type}")
        encoding = detect_file_encoding(file_path)
        logger.info(f"Detected encoding: {encoding}")
        
        # For binary files, read in binary mode
        if not file_type.startswith('text/'):
            with open(file_path, 'rb') as f:
                content = f.read()
            # Convert binary content to base64 string
            import base64
            content = base64.b64encode(content).decode('utf-8')
        else:
            # For text files, read with detected encoding
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
        
        # Determine file characteristics
        if file_type.startswith('text/'):
            # For text files, analyze the content
            lines = content.split('\n')
            avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
            has_comments = any(line.strip().startswith(('#', '//', '/*', '--')) for line in lines)
            
            # Try to detect the programming language
            if file_type == 'text/html':
                language = 'html'
                naming_convention = 'kebab-case'
            elif file_type == 'text/css':
                language = 'css'
                naming_convention = 'kebab-case'
            elif file_type == 'text/javascript':
                language = 'javascript'
                naming_convention = 'camelCase'
            elif file_type == 'text/x-python':
                language = 'python'
                naming_convention = 'snake_case'
            else:
                language = 'unknown'
                naming_convention = 'unknown'
        else:
            # For binary files, use generic characteristics
            language = 'binary'
            naming_convention = 'unknown'
            avg_line_length = 0
            has_comments = False

        training_data = {
            "code": content,
            "style": {
                "indentation": "spaces",
                "naming_convention": naming_convention,
                "line_length": "medium" if avg_line_length < 100 else "long",
                "comment_style": "present" if has_comments else "none"
            },
            "patterns": {
                "language": language,
                "file_type": file_type,
                "has_comments": has_comments,
                "is_binary": not file_type.startswith('text/')
            }
        }
        
        logger.info(f"Successfully converted file to training format")
        return training_data
    except Exception as e:
        logger.error(f"Error converting file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Digital Twin API is running"}

@app.post("/upload-data/")
async def upload_data(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        logger.info(f"Uploading file: {file.filename}")
        
        # Save the uploaded file
        file_path = f"uploaded_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get the file type
        content_type = file.content_type or mimetypes.guess_type(file.filename)[0] or 'application/octet-stream'
        logger.info(f"File type detected: {content_type}")
        
        # Convert the file to training format
        training_data = convert_file_to_training_format(file_path, content_type)
        
        # Save the converted data
        output_path = "sample_data.json"
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump([training_data], f, indent=2)
        
        # Validate the saved JSON
        if not validate_and_repair_json_file(output_path, "data"):
            raise HTTPException(status_code=500, detail="Failed to create valid training data file")
        
        # Clean up the temporary file
        os.remove(file_path)
        
        logger.info("File uploaded and converted successfully")
        return {"message": "Data uploaded and converted successfully", "status": "success"}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in upload_data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.post("/upload-config/")
async def upload_config(config: ConfigModel):
    try:
        logger.info("Uploading configuration")
        config_dict = config.dict()
        
        # Save the config
        with open("sample_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Validate the saved JSON
        if not validate_and_repair_json_file("sample_config.json", "config"):
            raise HTTPException(status_code=500, detail="Failed to create valid configuration file")
        
        logger.info("Configuration uploaded successfully")
        return {"message": "Config uploaded successfully", "status": "success"}
    except Exception as e:
        logger.error(f"Error in upload_config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/")
async def start_training(background_tasks: BackgroundTasks):
    if TRAIN_STATUS["status"] == "training":
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    def train_job():
        try:
            logger.info("Starting training process")
            TRAIN_STATUS["status"] = "training"
            TRAIN_STATUS["error"] = None
            
            # Check if config file exists and is valid
            if not validate_and_repair_json_file("sample_config.json", "config"):
                raise FileNotFoundError("Configuration file is missing, empty, or invalid. Please upload a valid configuration file.")
            # Check if data file exists and is valid
            if not validate_and_repair_json_file("sample_data.json", "data"):
                raise FileNotFoundError("Training data file is missing, empty, or invalid. Please upload valid training data.")
            
            # Load config
            with open("sample_config.json") as f:
                config = json.load(f)
            # Load and validate data
            with open("sample_data.json") as f:
                data = json.load(f)
                if not isinstance(data, list) or not data:
                    os.remove("sample_data.json")
                    raise ValueError("Training data file is invalid or empty. Please re-upload your data.")
            logger.info("Starting model training")
            data_path = "sample_data.json"
            output_path = "digital_twin_model.pth"
            losses = train_digital_twin(data_path, output_path, config)
            TRAIN_STATUS["status"] = "completed"
            TRAIN_STATUS["losses"] = losses
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            TRAIN_STATUS["status"] = "error"
            TRAIN_STATUS["error"] = str(e)
    
    background_tasks.add_task(train_job)
    return {"message": "Training started", "status": "success"}

@app.get("/status/")
async def get_status():
    return TRAIN_STATUS

@app.get("/download-model/")
async def download_model():
    model_path = "digital_twin_model.pth"
    if os.path.exists(model_path):
        return FileResponse(
            model_path,
            filename="digital_twin_model.pth",
            media_type="application/octet-stream"
        )
    raise HTTPException(status_code=404, detail="Model not found")

@app.post("/correct-code/", response_model=CodeCorrectionResponse)
async def correct_code(request: CodeCorrectionRequest):
    try:
        # Parse the code to check for syntax errors
        try:
            ast.parse(request.code)
        except SyntaxError as e:
            return CodeCorrectionResponse(
                original_code=request.code,
                corrected_code=request.code,
                suggestions=[{
                    "type": "syntax_error",
                    "message": str(e),
                    "line": e.lineno,
                    "offset": e.offset
                }],
                applied=False
            )

        # Apply code formatting
        formatted_code = autopep8.fix_code(request.code)
        
        # Apply Black formatting for consistent style
        formatted_code = black.format_str(formatted_code, mode=black.FileMode())
        
        # Generate suggestions
        suggestions = []
        
        # Check for common issues
        if len(formatted_code.splitlines()) > 0:
            # Check line length
            for i, line in enumerate(formatted_code.splitlines()):
                if len(line) > 88:  # Black's default line length
                    suggestions.append({
                        "type": "style",
                        "message": f"Line {i+1} exceeds recommended length",
                        "line": i+1,
                        "offset": 0
                    })
            
            # Check for missing docstrings in functions and classes
            tree = ast.parse(formatted_code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not ast.get_docstring(node):
                        suggestions.append({
                            "type": "documentation",
                            "message": f"Missing docstring for {node.name}",
                            "line": node.lineno,
                            "offset": 0
                        })

        # Auto-apply if requested
        if request.auto_apply:
            return CodeCorrectionResponse(
                original_code=request.code,
                corrected_code=formatted_code,
                suggestions=suggestions,
                applied=True
            )
        
        return CodeCorrectionResponse(
            original_code=request.code,
            corrected_code=formatted_code,
            suggestions=suggestions,
            applied=False
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 