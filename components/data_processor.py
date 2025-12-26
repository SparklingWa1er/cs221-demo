"""
Data processing utilities for LMCOR system.
Handles data loading, prompt construction, and preprocessing.
"""

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Tuple


class DataProcessor:
    """Process data for training and inference."""
    
    SEPARATOR = " <sep> "
    
    def __init__(self, task: str, mode: str):
        """
        Initialize data processor.
        
        Args:
            task: Task type ('MT' or 'GEC')
            mode: Mode type ('single' or 'multi')
        """
        self.task = task.upper()
        self.mode = mode.lower()
        
        if self.task not in ['MT', 'GEC']:
            raise ValueError(f"Task must be 'MT' or 'GEC', got {task}")
        if self.mode not in ['single', 'multi']:
            raise ValueError(f"Mode must be 'single' or 'multi', got {mode}")
    
    def construct_prompt(self, row: pd.Series) -> str:
        """
        Construct prompt from row data based on task and mode.
        
        Args:
            row: DataFrame row containing input and candidate columns
            
        Returns:
            Constructed prompt string
        """
        source = str(row.get('en', row.get('input', '')))
        
        if self.task == 'MT':
            # MT task uses format: "Rewrite: Source: {source} | Cand: {cand}" or multi format
            if self.mode == 'single':
                cand = str(row.get('candidate_1', ''))
                return f"Rewrite: Source: {source} | Cand: {cand}"
            else:  # multi
                c1 = str(row.get('candidate_1', ''))
                c2 = str(row.get('candidate_2', ''))
                c3 = str(row.get('candidate_3', ''))
                return f"Rewrite: Source: {source} | C1: {c1} | C2: {c2} | C3: {c3}"
        else:  # GEC
            # GEC task uses separator format
            if self.mode == 'single':
                cand = str(row.get('candidate_3', row.get('candidate_1', '')))
                return f"{source}{self.SEPARATOR}{cand}"
            else:  # multi
                c1 = str(row.get('candidate_1', ''))
                c2 = str(row.get('candidate_2', ''))
                c3 = str(row.get('candidate_3', ''))
                return f"{source}{self.SEPARATOR}{c1}{self.SEPARATOR}{c2}{self.SEPARATOR}{c3}"
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Loaded DataFrame
        """
        df = pd.read_csv(file_path)
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, tokenizer: AutoTokenizer, 
                             max_input_length: int = 1024, 
                             max_target_length: int = 256) -> Dataset:
        """
        Prepare data for training.
        
        Args:
            df: Input DataFrame
            tokenizer: Tokenizer instance
            max_input_length: Maximum input sequence length
            max_target_length: Maximum target sequence length
            
        Returns:
            Tokenized dataset
        """
        # Construct prompts
        df['input_text'] = df.apply(self.construct_prompt, axis=1)
        
        # Get target column (vi for MT, output for GEC)
        if self.task == 'MT':
            if 'vi' in df.columns:
                df['target_text'] = df['vi']
            elif 'output' in df.columns:
                df['target_text'] = df['output']
            else:
                raise ValueError("MT task requires 'vi' or 'output' column")
        else:  # GEC
            if 'output' in df.columns:
                df['target_text'] = df['output']
            elif 'vi' in df.columns:
                df['target_text'] = df['vi']
            else:
                raise ValueError("GEC task requires 'output' or 'vi' column")
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_pandas(df[['input_text', 'target_text']])
        
        # Split train/validation
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        # Tokenize
        def preprocess(examples):
            inputs = examples['input_text']
            targets = examples['target_text']
            
            model_inputs = tokenizer(
                inputs, 
                max_length=max_input_length, 
                truncation=True, 
                padding="max_length"
            )
            
            # Tokenize targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets, 
                    max_length=max_target_length, 
                    truncation=True, 
                    padding="max_length"
                )
            
            # Replace padding token id with -100 for loss calculation
            model_inputs["labels"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] 
                for label in labels["input_ids"]
            ]
            
            return model_inputs
        
        tokenized_data = dataset.map(preprocess, batched=True)
        return tokenized_data
    
    def prepare_inference_data(self, df: pd.DataFrame) -> List[str]:
        """
        Prepare data for inference.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of prompts
        """
        prompts = df.apply(self.construct_prompt, axis=1).tolist()
        return prompts

