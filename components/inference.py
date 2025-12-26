"""
Model inference component for LMCOR system.
"""

import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from .data_processor import DataProcessor


class ModelInference:
    """Perform inference with trained models."""
    
    def __init__(self, task: str = None, backbone: str = None, mode: str = None, 
                 model_dir: str = "./models", model_path: str = None):
        """
        Initialize inference.
        
        Args:
            task: Task type ('MT' or 'GEC') - required if model_path not provided
            backbone: Model backbone (e.g., 'google/mt5-base', 'VietAI/vit5-base') - required if model_path not provided
            mode: Mode type ('single' or 'multi') - required if model_path not provided
            model_dir: Base directory containing trained models (used if model_path not provided)
            model_path: Direct path to model directory (overrides task/backbone/mode)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Determine model path
        if model_path:
            # Use provided model path directly
            self.model_path = model_path
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            # Try to infer task, backbone, mode from metadata or path
            metadata_path = os.path.join(self.model_path, "training_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    self.task = metadata.get("task", "MT").upper()
                    self.backbone = metadata.get("backbone", "unknown")
                    self.mode = metadata.get("mode", "single").lower()
            else:
                # Infer from path structure
                path_parts = self.model_path.replace("\\", "/").split("/")
                if len(path_parts) >= 3:
                    self.task = path_parts[-3].upper() if path_parts[-3] in ["mt", "gec"] else "MT"
                    self.mode = path_parts[-1].lower() if path_parts[-1] in ["single", "multi"] else "single"
                    self.backbone = path_parts[-2].replace("_", "/") if len(path_parts) >= 2 else "unknown"
                else:
                    # Default values if cannot infer
                    self.task = task.upper() if task else "MT"
                    self.backbone = backbone if backbone else "unknown"
                    self.mode = mode.lower() if mode else "single"
        else:
            # Use task/backbone/mode to construct path
            if not all([task, backbone, mode]):
                raise ValueError("Either model_path or (task, backbone, mode) must be provided")
            
            self.task = task.upper()
            self.backbone = backbone
            self.mode = mode.lower()
            
            self.model_path = os.path.join(
                model_dir,
                self.task.lower(),
                backbone.replace("/", "_"),
                mode,
                "final_model"
            )
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model not found at {self.model_path}. Please train the model first."
                )
        
        # Initialize data processor
        self.data_processor = DataProcessor(self.task, self.mode)
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer from disk."""
        print(f"🔄 Loading model from {self.model_path}...")
        
        # Check if model files exist
        if not os.path.exists(os.path.join(self.model_path, "config.json")):
            raise FileNotFoundError(
                f"Model files not found at {self.model_path}. "
                f"Please ensure the model was saved correctly during training."
            )
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Load metadata if available
            metadata_path = os.path.join(self.model_path, "training_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                    print(f"📋 Model info: {self.metadata.get('task')} | {self.metadata.get('backbone')} | {self.metadata.get('mode')}")
            else:
                self.metadata = None
            
            print(f"✅ Model loaded successfully on {self.device}")
            print(f"   Task: {self.task}, Mode: {self.mode}, Backbone: {self.backbone}")
            
        except Exception as e:
            raise RuntimeError(f"Error loading model from {self.model_path}: {str(e)}")
    
    def predict(self, prompts: list, max_length: int = 256, 
                num_beams: int = 4) -> list:
        """
        Generate predictions for given prompts.
        
        Args:
            prompts: List of input prompts
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            
        Returns:
            List of generated texts
        """
        predictions = []
        
        with torch.no_grad():
            for prompt in tqdm(prompts, desc="Generating predictions"):
                try:
                    inputs = self.tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        max_length=1024, 
                        truncation=True
                    ).to(self.device)
                    
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=num_beams,
                        early_stopping=True
                    )
                    
                    pred_text = self.tokenizer.decode(
                        outputs[0], 
                        skip_special_tokens=True
                    )
                    predictions.append(pred_text)
                except Exception as e:
                    print(f"⚠️ Error during prediction: {e}")
                    # Return source text as fallback
                    predictions.append(prompt.split(self.data_processor.SEPARATOR)[0] 
                                     if self.data_processor.SEPARATOR in prompt 
                                     else prompt)
        
        return predictions
    
    def predict_file(self, test_file: str, output_file: str = None) -> pd.DataFrame:
        """
        Predict on test file and save results.
        
        Args:
            test_file: Path to test CSV file
            output_file: Path to save predictions (optional)
            
        Returns:
            DataFrame with predictions
        """
        # Load test data
        df = pd.read_csv(test_file)
        print(f"📊 Loaded {len(df)} test samples")
        
        # Prepare prompts
        prompts = self.data_processor.prepare_inference_data(df)
        
        # Generate predictions
        predictions = self.predict(prompts)
        
        # Add predictions to dataframe
        df['prediction'] = predictions
        
        # Save if output file specified
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(output_file, index=False)
            print(f"💾 Predictions saved to {output_file}")
        
        return df

