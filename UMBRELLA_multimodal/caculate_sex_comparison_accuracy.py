import json
from collections import Counter, defaultdict
import os

def calculate_sex_classification_accuracy(file_path):
    """
    prediction_results.jsonl 파일을 읽어 Subject별 Majority Vote 정확도를 계산합니다.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # 데이터 저장소: {subject_id: {'gt': label, 'preds': [pred1, pred2, ...]}}
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

                # 필수 정보가 없으면 스킵
                if not subject_id or not subject_label:
                    continue

                # Prediction 전처리 (소문자 변환, 구두점 제거)
                # 예: "Male." -> "male", "female" -> "female"
                clean_pred = raw_prediction.strip().lower().replace('.', '')
                
                # 'male' 또는 'female'이 포함되어 있는지 확인하여 정규화
                if 'female' in clean_pred:
                    final_pred_token = 'female'
                elif 'male' in clean_pred:
                    final_pred_token = 'male'
                else:
                    final_pred_token = None # 알 수 없는 예측

                # 데이터 저장
                if subject_data[subject_id]['gt'] is None:
                    subject_data[subject_id]['gt'] = subject_label
                
                if final_pred_token:
                    subject_data[subject_id]['preds'].append(final_pred_token)
                    
            except json.JSONDecodeError:
                continue

    # 정확도 계산
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

        # [핵심] Majority Vote: 가장 빈도가 높은 예측값 선택
        # most_common(1)은 [(값, 개수)] 형태의 리스트를 반환
        counter = Counter(predictions)
        most_common = counter.most_common(1)
        final_prediction = most_common[0][0] # 가장 많이 나온 예측값
        
        # 정확도 판별
        is_correct = (final_prediction == ground_truth)
        if is_correct:
            correct_subjects += 1
        
        total_subjects += 1
        
        # 결과 출력 (상위 5개만 예시로 보려면 슬라이싱 가능, 여기서는 전체 출력)
        # 딕셔너리 형태의 투표 결과 문자열 생성 (예: {'female': 5, 'male': 1})
        vote_str = str(dict(counter))
        print(f"{subject_id:<20} | {ground_truth:<10} | {final_prediction:<15} | {vote_str}")

    # 최종 결과 출력
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
    # 여기에 결과 파일 경로를 입력하세요
    result_file_path = "/pscratch/sd/h/heehaw/BrainVLM/UMBRELLA_gemini/eval_predictions_SexMixed_100subjects_20samples/comparison_prediction_results_ckpt400.jsonl" 
    calculate_sex_classification_accuracy(result_file_path)