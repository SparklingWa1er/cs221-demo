"""
Model training component for LMCOR system.
"""

import os
import json
import pandas as pd
import torch
import gc
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq
)

# Import Seq2SeqTrainer - try multiple import methods
try:
    # Standard import
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
except (ImportError, ModuleNotFoundError):
    try:
        # Try importing from trainer_seq2seq module directly
        from transformers.trainer_seq2seq import Seq2SeqTrainer, Seq2SeqTrainingArguments
    except (ImportError, ModuleNotFoundError):
        # Last resort: try Trainer with seq2seq capabilities
        from transformers import Trainer, TrainingArguments
        # Create aliases - note: this may not work perfectly but allows code to run
        Seq2SeqTrainer = Trainer
        Seq2SeqTrainingArguments = TrainingArguments
        import warnings
        warnings.warn(
            "Seq2SeqTrainer not found. Using Trainer instead. "
            "Some seq2seq-specific features may not work. "
            "Please upgrade transformers: pip install --upgrade transformers>=4.30.0"
        )
from datasets import Dataset
from .data_processor import DataProcessor


class ModelTrainer:
    """Train LMCOR models."""
    
    def __init__(self, task: str, backbone: str, mode: str, 
                 output_dir: str = "./models"):
        """
        Initialize trainer.
        
        Args:
            task: Task type ('MT' or 'GEC')
            backbone: Model backbone (e.g., 'google/mt5-base', 'VietAI/vit5-base')
            mode: Mode type ('single' or 'multi')
            output_dir: Base directory for saving models
        """
        self.task = task.upper()
        self.backbone = backbone
        self.mode = mode.lower()
        self.output_dir = output_dir
        self.data_processor = DataProcessor(task, mode)
        
        # Determine batch size based on model size
        if "large" in backbone.lower():
            self.batch_size = 2
        else:
            self.batch_size = 16
        
        # Model output directory
        self.model_dir = os.path.join(
            output_dir, 
            self.task.lower(),
            backbone.replace("/", "_"),
            mode
        )
        os.makedirs(self.model_dir, exist_ok=True)
    
    def train(self, train_file: str, 
              epochs: int = 3,
              learning_rate: float = 3e-4,
              max_input_length: int = 1024,
              max_target_length: int = 256,
              save_checkpoints: bool = True,
              save_total_limit: int = 3):
        """
        Train the model.
        
        Args:
            train_file: Path to training CSV file
            epochs: Number of training epochs
            learning_rate: Learning rate
            max_input_length: Maximum input sequence length
            max_target_length: Maximum target sequence length
            save_checkpoints: Whether to save checkpoints during training
            save_total_limit: Maximum number of checkpoints to keep
            
        Returns:
            Path to saved model
        """
        # Model paths
        model_path = os.path.join(self.model_dir, "final_model")
        checkpoint_dir = os.path.join(self.model_dir, "checkpoints")
        
        # Check if model already exists
        if os.path.exists(model_path):
            print(f"⏩ Model already exists at {model_path}, skipping training.")
            return model_path
        
        print(f"\n{'='*40}")
        print(f"🚀 TRAINING: {self.task} | {self.backbone} | {self.mode.upper()}")
        print(f"⚙️  Batch: {self.batch_size} | BF16: On")
        print(f"📁 Model will be saved to: {model_path}")
        if save_checkpoints:
            print(f"📁 Checkpoints will be saved to: {checkpoint_dir}")
        print(f"{'='*40}")
        
        # Load data
        df = pd.read_csv(train_file)
        print(f"📊 Loaded {len(df)} training samples")
        
        # Load tokenizer and model
        print("🔄 Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(self.backbone)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.backbone)
        print("✅ Model loaded")
        
        # Prepare data
        print("🔄 Preparing training data...")
        tokenized_data = self.data_processor.prepare_training_data(
            df, tokenizer, max_input_length, max_target_length
        )
        print(f"✅ Prepared {len(tokenized_data['train'])} train and {len(tokenized_data['test'])} eval samples")
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=checkpoint_dir if save_checkpoints else self.model_dir,
            eval_strategy="epoch",
            save_strategy="epoch" if save_checkpoints else "no",
            save_total_limit=save_total_limit if save_checkpoints else 1,
            learning_rate=learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=epochs,
            predict_with_generate=True,
            fp16=False,
            bf16=True,  # Use BF16 for A100 or compatible GPUs
            logging_steps=50,
            report_to="none",
            load_best_model_at_end=True if save_checkpoints else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer, 
            model=model
        )
        
        # Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["test"],
            data_collator=data_collator,
            processing_class=tokenizer,
        )
        
        # Train
        print("🔄 Starting training...")
        train_result = trainer.train()
        
        # Save final model
        print(f"💾 Saving final model to {model_path}...")
        os.makedirs(model_path, exist_ok=True)
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        
        # Save training metadata
        metadata = {
            "task": self.task,
            "backbone": self.backbone,
            "mode": self.mode,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": self.batch_size,
            "max_input_length": max_input_length,
            "max_target_length": max_target_length,
            "train_samples": len(tokenized_data["train"]),
            "eval_samples": len(tokenized_data["test"]),
            "train_file": train_file,
            "final_train_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else None,
        }
        
        metadata_path = os.path.join(model_path, "training_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"💾 Training metadata saved to {metadata_path}")
        
        # Cleanup
        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"✅ Training completed! Model saved at {model_path}")
        if save_checkpoints:
            print(f"📁 Checkpoints saved at {checkpoint_dir}")
        
        return model_path

