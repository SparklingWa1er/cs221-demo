"""
Main script for LMCOR training and inference system.
"""

import argparse
import os
import datetime
from pathlib import Path
import pandas as pd

from components import ModelTrainer, ModelInference, ModelEvaluator


def get_output_paths(task: str, backbone: str, mode: str, test_file: str):
    """
    Generate output paths for predictions and reports.
    
    Args:
        task: Task type ('MT' or 'GEC')
        backbone: Model backbone
        mode: Mode type ('single' or 'multi')
        test_file: Path to test file
        
    Returns:
        Tuple of (prediction_path, report_path)
    """
    # Get test file name without extension
    test_name = Path(test_file).stem
    
    # Normalize backbone name (replace / with _)
    backbone_norm = backbone.replace("/", "_")
    
    # Prediction path: task/prediction/backbone/mode/test_name.csv
    pred_dir = os.path.join(task.lower(), "prediction", backbone_norm, mode)
    pred_path = os.path.join(pred_dir, f"{test_name}.csv")
    
    # Report path: task/report/backbone/mode/test_name.txt
    report_dir = os.path.join(task.lower(), "report", backbone_norm, mode)
    report_path = os.path.join(report_dir, f"{test_name}.txt")
    
    return pred_path, report_path


def save_report(report_path: str, task: str, backbone: str, mode: str,
                test_file: str, metrics: dict, candidate_metrics: dict = None):
    """
    Save evaluation report to text file.
    
    Args:
        report_path: Path to save report
        task: Task type
        backbone: Model backbone
        mode: Mode type
        test_file: Test file path
        metrics: Main evaluation metrics
        candidate_metrics: Metrics for candidates (optional)
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(f"📊 BÁO CÁO KẾT QUẢ THỰC NGHIỆM LMCOR")
    report_lines.append(f"🕒 Thời gian: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 60)
    report_lines.append(f"\n=== THÔNG TIN THỰC NGHIỆM ===")
    report_lines.append(f"Task: {task}")
    report_lines.append(f"Backbone: {backbone}")
    report_lines.append(f"Mode: {mode}")
    report_lines.append(f"Test File: {test_file}")
    
    report_lines.append(f"\n=== KẾT QUẢ ĐÁNH GIÁ ===")
    
    if task == 'MT':
        report_lines.append(f"BLEU Score: {metrics.get('BLEU', 'N/A')}")
        if metrics.get('Baseline_BLEU'):
            report_lines.append(f"Baseline BLEU: {metrics.get('Baseline_BLEU', 'N/A')}")
            report_lines.append(f"Gain: {metrics.get('Gain', 'N/A')}")
    else:  # GEC
        report_lines.append(f"Precision: {metrics.get('Precision', 'N/A')}")
        report_lines.append(f"Recall: {metrics.get('Recall', 'N/A')}")
        report_lines.append(f"F0.5: {metrics.get('F0.5', 'N/A')}")
    
    if candidate_metrics:
        report_lines.append(f"\n=== SO SÁNH VỚI CANDIDATES ===")
        if task == 'MT':
            report_lines.append(f"{'Model/Candidate':<20} | {'BLEU':<10}")
            report_lines.append("-" * 35)
            for name, m in candidate_metrics.items():
                report_lines.append(f"{name:<20} | {m.get('BLEU', 'N/A'):<10}")
        else:  # GEC
            report_lines.append(f"{'Model/Candidate':<20} | {'Precision':<10} | {'Recall':<10} | {'F0.5':<10}")
            report_lines.append("-" * 60)
            for name, m in candidate_metrics.items():
                report_lines.append(
                    f"{name:<20} | {m.get('Precision', 'N/A'):<10} | "
                    f"{m.get('Recall', 'N/A'):<10} | {m.get('F0.5', 'N/A'):<10}"
                )
    
    report_content = "\n".join(report_lines)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"📄 Report saved to {report_path}")


def train_model(args):
    """Train a model."""
    trainer = ModelTrainer(
        task=args.task,
        backbone=args.backbone,
        mode=args.mode,
        output_dir=args.model_dir
    )
    
    trainer.train(
        train_file=args.train_file,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )


def infer_model(args):
    """Run inference on test data."""
    # Initialize inference
    if hasattr(args, 'model_path') and args.model_path:
        # Use direct model path
        inference = ModelInference(model_path=args.model_path)
        # Extract task, backbone, mode from inference object for output paths
        task = inference.task
        backbone = inference.backbone
        mode = inference.mode
    else:
        # Use task/backbone/mode to find model
        inference = ModelInference(
            task=args.task,
            backbone=args.backbone,
            mode=args.mode,
            model_dir=args.model_dir
        )
        task = args.task
        backbone = args.backbone
        mode = args.mode
    
    # Get output paths
    pred_path, report_path = get_output_paths(
        task, backbone, mode, args.test_file
    )
    
    # Run inference
    df_results = inference.predict_file(args.test_file, pred_path)
    
    # Evaluate (use task from inference object)
    evaluator = ModelEvaluator(task=task)
    metrics = evaluator.evaluate(df_results)
    
    # Evaluate candidates for comparison
    candidate_metrics = None
    if all(col in df_results.columns for col in ['candidate_1', 'candidate_2', 'candidate_3']):
        candidate_metrics = evaluator.evaluate_candidates(df_results)
    
    # Save report
    save_report(
        report_path, task, backbone, mode,
        args.test_file, metrics, candidate_metrics
    )
    
    print(f"\n✅ Inference completed!")
    print(f"📊 Predictions: {pred_path}")
    print(f"📄 Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="LMCOR Training and Inference System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python main.py train --task MT --backbone google/mt5-base --mode single --train_file data/train.csv
  
  # Run inference
  python main.py infer --task MT --backbone google/mt5-base --mode single --test_file data/test.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--task', type=str, required=True, 
                             choices=['MT', 'GEC'],
                             help='Task type: MT (Machine Translation) or GEC (Grammar Error Correction)')
    train_parser.add_argument('--backbone', type=str, required=True,
                             help='Model backbone (e.g., google/mt5-base, VietAI/vit5-base)')
    train_parser.add_argument('--mode', type=str, required=True,
                             choices=['single', 'multi'],
                             help='Mode: single (1 candidate) or multi (3 candidates)')
    train_parser.add_argument('--train_file', type=str, required=True,
                             help='Path to training CSV file')
    train_parser.add_argument('--model_dir', type=str, default='./models',
                             help='Directory to save models (default: ./models)')
    train_parser.add_argument('--epochs', type=int, default=3,
                             help='Number of training epochs (default: 3)')
    train_parser.add_argument('--learning_rate', type=float, default=3e-4,
                             help='Learning rate (default: 3e-4)')
    
    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--model_path', type=str, default=None,
                            help='Direct path to trained model directory (if provided, overrides task/backbone/mode)')
    infer_parser.add_argument('--task', type=str, default=None,
                            choices=['MT', 'GEC'],
                            help='Task type: MT or GEC (required if model_path not provided)')
    infer_parser.add_argument('--backbone', type=str, default=None,
                             help='Model backbone (e.g., google/mt5-base, VietAI/vit5-base) (required if model_path not provided)')
    infer_parser.add_argument('--mode', type=str, default=None,
                             choices=['single', 'multi'],
                             help='Mode: single or multi (required if model_path not provided)')
    infer_parser.add_argument('--test_file', type=str, required=True,
                             help='Path to test CSV file')
    infer_parser.add_argument('--model_dir', type=str, default='./models',
                             help='Directory containing trained models (default: ./models, used if model_path not provided)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'infer':
        # Validate arguments for inference
        if not args.model_path:
            if not all([args.task, args.backbone, args.mode]):
                infer_parser.error("Either --model_path or (--task, --backbone, --mode) must be provided")
        infer_model(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

