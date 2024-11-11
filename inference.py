import argparse
import os
import json
import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoConfig, BertTokenizerFast
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from tqdm import tqdm
from typing import Dict, List, Union
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support
import pathlib
from collections import defaultdict, Counter

def calculate_label_metrics(all_predictions: List[str], all_labels: List[str]) -> str:
    """
    Calculate precision, recall, and F1 score for each label based only on rows where that label appears,
    formatted exactly like sklearn's classification_report.
    """
    # Get unique labels (excluding 'O')
    unique_labels = sorted(set(label for label in all_labels + all_predictions if label != 'O'))
    
    # Calculate metrics for each label
    results = []
    for label in unique_labels:
        # Create binary arrays for current label
        true_binary = np.array([1 if l == label else 0 for l in all_labels])
        pred_binary = np.array([1 if p == label else 0 for p in all_predictions])
        
        # Calculate metrics for this label
        precision, recall, f1, support = precision_recall_fscore_support(
            true_binary, 
            pred_binary, 
            average='binary',
            zero_division="warn"
        )
        
        results.append({
            'label': label,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': np.sum(true_binary)  # Count of actual occurrences
        })
    
    # Format output like sklearn's classification_report
    report_lines = []
    report_lines.append("                    precision    recall  f1-score   support")
    report_lines.append("")
    
    # Add per-label metrics
    for res in results:
        label_line = f"{res['label']:>20} {res['precision']:>9.4f} {res['recall']:>8.4f} {res['f1']:>8.4f} {res['support']:>9d}"
        report_lines.append(label_line)
    
    # Calculate and add averages
    total_support = sum(res['support'] for res in results)
    if total_support > 0:
        # Micro average (calculate metrics globally)
        all_true_binary = np.array([1 if l != 'O' else 0 for l in all_labels])
        all_pred_binary = np.array([1 if p != 'O' else 0 for p in all_predictions])
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
            all_true_binary, 
            all_pred_binary, 
            average='binary',
            zero_division="warn"
        )
        
        # Macro average (unweighted mean of per-label metrics)
        macro_p = np.mean([res['precision'] for res in results])
        macro_r = np.mean([res['recall'] for res in results])
        macro_f1 = np.mean([res['f1'] for res in results])
        
        # Weighted average (weighted by support)
        weighted_p = np.average([res['precision'] for res in results], 
                              weights=[res['support'] for res in results])
        weighted_r = np.average([res['recall'] for res in results], 
                              weights=[res['support'] for res in results])
        weighted_f1 = np.average([res['f1'] for res in results], 
                               weights=[res['support'] for res in results])
        
        report_lines.append("")
        report_lines.append(f"{'micro avg':>20} {micro_p:>9.4f} {micro_r:>8.4f} {micro_f1:>8.4f} {total_support:>9d}")
        report_lines.append(f"{'macro avg':>20} {macro_p:>9.4f} {macro_r:>8.4f} {macro_f1:>8.4f} {total_support:>9d}")
        report_lines.append(f"{'weighted avg':>20} {weighted_p:>9.4f} {weighted_r:>8.4f} {weighted_f1:>8.4f} {total_support:>9d}")
    
    return '\n'.join(report_lines)

class LabelMapper:
    def __init__(self):
        self.label_to_id: Dict[str, int] = {'O': 0}
        self.id_to_label: Dict[int, str] = {0: 'O'}
        self.num_labels: int = 1
        
    def decode(self, ids: List[int]) -> List[str]:
        decoded_labels = []
        prev_id = 0
        
        for id in ids:
            if id == 0:
                decoded_labels.append('O')
                prev_id = 0
            else:
                base_label = self.id_to_label[id][2:]
                if id != prev_id:
                    decoded_labels.append(f'B-{base_label}')
                else:
                    decoded_labels.append(f'I-{base_label}')
                prev_id = id
                
        return decoded_labels

def process_batch(batch_texts: List[str], batch_labels: List[Union[str, List]], 
                 tokenizer, model, label_mapper, device) -> tuple[List[str], List[str]]:
    """Process a single batch of texts and return predictions and true labels."""
    try:
        # Filter out any None or empty values
        valid_pairs = [(text, labels) for text, labels in zip(batch_texts, batch_labels) 
                      if text and labels and isinstance(text, str)]
        
        if not valid_pairs:
            return [], []
            
        filtered_texts, filtered_labels = zip(*valid_pairs)
        
        # Tokenize texts
        encoded = tokenizer(
            filtered_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {
            'input_ids': encoded['input_ids'].to(device),
            'attention_mask': encoded['attention_mask'].to(device)
        }
        
        # Get model predictions
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        
        batch_predictions = []
        batch_true_labels = []
        
        # Process each sequence in the batch
        for i, (pred, text, label) in enumerate(zip(predictions, filtered_texts, filtered_labels)):
            # Get valid predictions (non-padding)
            valid_length = min(len(tokenizer.tokenize(text)) + 2, 128)  # +2 for [CLS] and [SEP]
            valid_pred = pred[:valid_length].cpu().tolist()
            
            # Decode predictions to labels
            pred_labels = label_mapper.decode(valid_pred)
            
            # Process true labels
            if isinstance(label, str):
                try:
                    true_labels = eval(label)
                except:
                    print(f"Warning: Could not evaluate label string: {label}")
                    continue
            else:
                true_labels = label
                
            # Ensure labels are properly truncated
            true_labels = true_labels[:valid_length]
            pred_labels = pred_labels[:valid_length]
            
            # Only add if lengths match
            if len(true_labels) == len(pred_labels):
                batch_predictions.extend(pred_labels)
                batch_true_labels.extend(true_labels)
            
        return batch_predictions, batch_true_labels
        
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        return [], []
    

def load_model(model_path: str, device: torch.device):
    """Load the trained model and required components."""
    print("Loading model and components...")
    
    try:
        # Load config
        config = AutoConfig.from_pretrained(os.path.join(model_path, "checkpoints/best_model"))
        
        # Initialize model with config
        model = AutoModelForTokenClassification.from_config(config)
        
        # Load model weights
        state_dict = load_file(os.path.join(model_path, "checkpoints/best_model/model.safetensors"))
        model.load_state_dict(state_dict)
        
        # Move model to device
        model = model.to(device)
        model.eval()
        
        # Load tokenizer
        tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-multilingual-cased")
        
        # Load label mapper
        with open(os.path.join(model_path, "label_mapper.json"), 'r') as f:
            label_mapper_data = json.load(f)
        
        label_mapper = LabelMapper()
        label_mapper.label_to_id = label_mapper_data['label_to_id']
        label_mapper.id_to_label = {int(k): v for k, v in label_mapper_data['id_to_label'].items()}
        label_mapper.num_labels = label_mapper_data['num_labels']
        
        return model, tokenizer, label_mapper
    
    except Exception as e:
        raise RuntimeError(f"Error loading model components: {str(e)}")

def load_test_data(file_path: str) -> pd.DataFrame:
    """Load and validate test data from various file formats."""
    print(f"Loading test data from {file_path}")
    
    file_extension = pathlib.Path(file_path).suffix.lower()
    
    try:
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension == '.jsonl':
            df = pd.read_json(file_path,lines=True)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_extension == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Verify required columns exist
        required_columns = ['source_text', 'mbert_token_classes']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Remove rows with None/NaN values
        initial_len = len(df)
        df = df.dropna(subset=['source_text', 'mbert_token_classes'])
        if len(df) < initial_len:
            print(f"Removed {initial_len - len(df)} rows with missing values")
        
        return df
    
    except Exception as e:
        raise RuntimeError(f"Error loading test data: {str(e)}")

def evaluate_model(model, tokenizer, label_mapper, test_data, device, batch_size=32):
    """Evaluate model on test data and return metrics."""
    print("\nStarting model evaluation...")
    
    all_predictions = []
    all_labels = []
    error_count = 0
    processed_count = 0
    
    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(range(0, len(test_data), batch_size))
        for idx in progress_bar:
            batch_texts = test_data['source_text'].iloc[idx:idx + batch_size].tolist()
            batch_labels = test_data['mbert_token_classes'].iloc[idx:idx + batch_size].tolist()
            
            batch_predictions, batch_true_labels = process_batch(
                batch_texts, batch_labels, tokenizer, model, label_mapper, device
            )
            
            if batch_predictions and batch_true_labels:
                all_predictions.extend(batch_predictions)
                all_labels.extend(batch_true_labels)
                processed_count += 1
            else:
                error_count += 1
            
            progress_bar.set_description(
                f"Processed: {processed_count} batches, Errors: {error_count}"
            )
    
    print(f"\nProcessing complete. Successfully processed {processed_count} batches with {error_count} errors.")
    
    if not all_predictions or not all_labels:
        raise ValueError("No valid predictions were generated. Check the input data and model.")
    
    if len(all_predictions) != len(all_labels):
        print(f"Warning: Mismatch in predictions ({len(all_predictions)}) and labels ({len(all_labels)})")
        min_len = min(len(all_predictions), len(all_labels))
        all_predictions = all_predictions[:min_len]
        all_labels = all_labels[:min_len]
    
    # Calculate metrics using the updated function
    detailed_report = calculate_label_metrics(all_predictions, all_labels)
    
    return {
        'detailed_report': detailed_report,
        'processing_stats': {
            'total_batches': processed_count + error_count,
            'successful_batches': processed_count,
            'error_batches': error_count,
            'total_predictions': len(all_predictions),
            'unique_labels': len(set(label for label in all_labels if label != 'O'))
        }
    }



def main():
    parser = argparse.ArgumentParser(description='Evaluate PII detection model on test data')
    parser.add_argument('-s', '--source', required=True, help='Path to test data file')
    parser.add_argument('-m', '--model', default='pii_model', help='Path to model directory')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('-o', '--output', help='Custom path for output results')
    args = parser.parse_args()
    
    try:
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load and validate test data
        test_data = load_test_data(args.source)
        print(f"Loaded {len(test_data)} valid test samples")
        
        # Load model and components
        model, tokenizer, label_mapper = load_model(args.model, device)
        
        # Evaluate model
        results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            label_mapper=label_mapper,
            test_data=test_data,
            device=device,
            batch_size=args.batch_size
        )
        
        # Print results
        print("\n=== Model Performance Metrics ===")
        print("\nProcessing Statistics:")
        for key, value in results['processing_stats'].items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        print("\nDetailed Classification Report:")
        print(results['detailed_report'])
        
        # Save results
        output_dir = args.output if args.output else os.path.join(args.model, 'evaluation_results')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(output_dir, f'test_evaluation_results_{timestamp}.json')
        
        with open(results_path, 'w') as f:
            json.dump({
                'test_file': args.source,
                'processing_stats': results['processing_stats'],
                'detailed_report': results['detailed_report']
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()