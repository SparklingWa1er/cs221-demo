"""
Model evaluation component for LMCOR system.
Supports BLEU for MT task and ERRANT for GEC task.
"""

import pandas as pd
from typing import Dict, List, Optional
from sacrebleu.metrics import BLEU
from tqdm import tqdm

# Try to import ERRANT for GEC evaluation
try:
    import errant
    import spacy
    import sys
    import subprocess
    ERRANT_AVAILABLE = True
except ImportError:
    ERRANT_AVAILABLE = False
    print("⚠️ ERRANT not available. GEC evaluation will use BLEU instead.")


class ModelEvaluator:
    """Evaluate model predictions."""
    
    def __init__(self, task: str):
        """
        Initialize evaluator.
        
        Args:
            task: Task type ('MT' or 'GEC')
        """
        self.task = task.upper()
        self.errant_annotator = None
        
        if self.task == 'GEC' and ERRANT_AVAILABLE:
            self._ensure_spacy_model()
            try:
                self.errant_annotator = errant.load('en')
            except Exception as e:
                print(f"⚠️ Could not load ERRANT: {e}. Will use BLEU instead.")
                self.task = 'MT'  # Fallback to BLEU
    
    def _ensure_spacy_model(self, model_name: str = 'en_core_web_sm'):
        """Ensure spaCy model is installed."""
        try:
            spacy.load(model_name)
        except OSError:
            print(f"⚠️ Installing {model_name}...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
    
    def evaluate_mt(self, df: pd.DataFrame, pred_col: str = 'prediction',
                    ref_col: str = 'vi') -> Dict[str, float]:
        """
        Evaluate MT task using BLEU score.
        
        Args:
            df: DataFrame with predictions and references
            pred_col: Column name for predictions
            ref_col: Column name for references
            
        Returns:
            Dictionary with BLEU scores
        """
        predictions = df[pred_col].astype(str).tolist()
        references = df[ref_col].astype(str).tolist()
        
        # Calculate BLEU
        bleu_metric = BLEU(trg_lang="vi")
        bleu_score = bleu_metric.corpus_score(predictions, [references]).score
        
        # Calculate baseline BLEU (best of candidates)
        if all(col in df.columns for col in ['candidate_1', 'candidate_2', 'candidate_3']):
            b1 = bleu_metric.corpus_score(
                df['candidate_1'].astype(str).tolist(), [references]
            ).score
            b2 = bleu_metric.corpus_score(
                df['candidate_2'].astype(str).tolist(), [references]
            ).score
            b3 = bleu_metric.corpus_score(
                df['candidate_3'].astype(str).tolist(), [references]
            ).score
            best_baseline = max(b1, b2, b3)
            gain = bleu_score - best_baseline
        else:
            best_baseline = None
            gain = None
        
        results = {
            'BLEU': round(bleu_score, 2),
            'Baseline_BLEU': round(best_baseline, 2) if best_baseline else None,
            'Gain': round(gain, 2) if gain is not None else None
        }
        
        return results
    
    def evaluate_gec(self, df: pd.DataFrame, pred_col: str = 'prediction',
                     src_col: str = 'input', ref_col: str = 'output') -> Dict[str, float]:
        """
        Evaluate GEC task using ERRANT (Precision, Recall, F0.5).
        
        Args:
            df: DataFrame with predictions, sources, and references
            pred_col: Column name for predictions
            src_col: Column name for source texts
            ref_col: Column name for reference texts
            
        Returns:
            Dictionary with Precision, Recall, F0.5 scores
        """
        if self.errant_annotator is None:
            # Fallback to BLEU if ERRANT not available
            print("⚠️ Using BLEU instead of ERRANT for GEC evaluation")
            return self.evaluate_mt(df, pred_col, ref_col)
        
        print("⏳ Evaluating with ERRANT (this may take a while)...")
        tp, fp, fn = 0, 0, 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            src = str(row[src_col])
            hyp = str(row[pred_col])
            ref = str(row[ref_col])
            
            if pd.isna(hyp):
                hyp = ""
            
            try:
                # Parse texts
                orig = self.errant_annotator.parse(src)
                cor = self.errant_annotator.parse(hyp)
                gold = self.errant_annotator.parse(ref)
                
                # Annotate edits
                hyp_edits = self.errant_annotator.annotate(orig, cor)
                gold_edits = self.errant_annotator.annotate(orig, gold)
                
                # Convert to sets for comparison
                hyp_set = set([(e.o_start, e.o_end, e.c_str) for e in hyp_edits])
                gold_set = set([(e.o_start, e.o_end, e.c_str) for e in gold_edits])
                
                # Calculate TP, FP, FN
                tp += len(hyp_set & gold_set)
                fp += len(hyp_set - gold_set)
                fn += len(gold_set - hyp_set)
            except Exception as e:
                print(f"⚠️ Error processing row: {e}")
                continue
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        beta = 0.5
        f05 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0
        
        results = {
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F0.5': round(f05, 4)
        }
        
        return results
    
    def evaluate(self, df: pd.DataFrame, pred_col: str = 'prediction') -> Dict[str, float]:
        """
        Evaluate predictions based on task type.
        
        Args:
            df: DataFrame with predictions
            pred_col: Column name for predictions
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.task == 'MT':
            return self.evaluate_mt(df, pred_col)
        else:  # GEC
            return self.evaluate_gec(df, pred_col)
    
    def evaluate_candidates(self, df: pd.DataFrame, 
                           candidate_cols: List[str] = ['candidate_1', 'candidate_2', 'candidate_3'],
                           pred_col: str = 'prediction') -> Dict[str, Dict[str, float]]:
        """
        Evaluate predictions and all candidates for comparison.
        
        Args:
            df: DataFrame with predictions and candidates
            candidate_cols: List of candidate column names
            pred_col: Column name for predictions
            
        Returns:
            Dictionary mapping column names to evaluation results
        """
        results = {}
        
        # Evaluate prediction
        results[pred_col] = self.evaluate(df, pred_col)
        
        # Evaluate each candidate
        for col in candidate_cols:
            if col in df.columns:
                # Create temporary dataframe with candidate as prediction
                temp_df = df.copy()
                temp_df['temp_pred'] = temp_df[col]
                results[col] = self.evaluate(temp_df, 'temp_pred')
        
        return results

