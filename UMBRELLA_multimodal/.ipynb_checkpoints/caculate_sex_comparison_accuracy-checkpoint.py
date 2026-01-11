import json
from collections import Counter, defaultdict
import os

def calculate_sex_classification_accuracy(file_path):
    """
    Read prediction_results.jsonl and calculate per-subject Majority Vote accuracy.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # Data storage: {subject_id: {'gt': label, 'preds': [pred1, pred2, ...]}}
    subject_data = defaultdict(lambda: {'gt': None, 'preds': []})
    
    print(f"Loading results from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                metadata = item.get('metadata', {})
                
                subject_id = metadata.get('subject_id')
                subject_label = metadata.get('subject_label')
                raw_prediction = item.get('model_prediction', '')

                # Skip if required info is missing
                if not subject_id or not subject_label:
                    continue

                # Prediction preprocessing (lowercase, remove punctuation)
                # e.g.: "Male." -> "male", "female" -> "female"
                clean_pred = raw_prediction.strip().lower().replace('.', '')
                
                # Normalize by checking if 'male' or 'female' is included
                if 'female' in clean_pred:
                    final_pred_token = 'female'
                elif 'male' in clean_pred:
                    final_pred_token = 'male'
                else:
                    final_pred_token = None  # Unknown prediction

                # Store data
                if subject_data[subject_id]['gt'] is None:
                    subject_data[subject_id]['gt'] = subject_label
                
                if final_pred_token:
                    subject_data[subject_id]['preds'].append(final_pred_token)
                    
            except json.JSONDecodeError:
                continue

    # Calculate accuracy
    total_subjects = 0
    correct_subjects = 0
    no_prediction_subjects = 0

    print("\n" + "="*50)
    print(f"{'Subject ID':<20} | {'GT':<10} | {'Vote Result':<15} | {'Prediction Counts'}")
    print("-" * 50)

    for subject_id, data in subject_data.items():
        ground_truth = data['gt']
        predictions = data['preds']
        
        if not predictions:
            no_prediction_subjects += 1
            print(f"{subject_id:<20} | {ground_truth:<10} | {'None':<15} | (No valid preds)")
            continue

        # [Key] Majority Vote: Select the most frequent prediction
        # most_common(1) returns a list of [(value, count)]
        counter = Counter(predictions)
        most_common = counter.most_common(1)
        final_prediction = most_common[0][0]  # Most frequent prediction
        
        # Determine accuracy
        is_correct = (final_prediction == ground_truth)
        if is_correct:
            correct_subjects += 1
        
        total_subjects += 1
        
        # Print results (can slice for top 5, here prints all)
        # Generate vote result string as dict (e.g.: {'female': 5, 'male': 1})
        vote_str = str(dict(counter))
        print(f"{subject_id:<20} | {ground_truth:<10} | {final_prediction:<15} | {vote_str}")

    # Print final results
    accuracy = (correct_subjects / total_subjects) * 100 if total_subjects > 0 else 0.0
    
    print("=" * 50)
    print(f"Total Subjects: {len(subject_data)}")
    print(f"Valid Subjects (with preds): {total_subjects}")
    print(f"Correctly Classified: {correct_subjects}")
    print(f"Skipped (No preds): {no_prediction_subjects}")
    print("-" * 50)
    print(f"Final Accuracy (Majority Vote): {accuracy:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    # Enter the result file path here
    result_file_path = "/pscratch/sd/h/heehaw/BrainVLM/UMBRELLA_gemini/eval_predictions_SexMixed_100subjects_20samples/comparison_prediction_results_ckpt400.jsonl" 
    calculate_sex_classification_accuracy(result_file_path)