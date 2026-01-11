import re

def extract_label(text):
    # Define pattern to find
    # Find word after "appears to be a " and before " subject."
    pattern = r"appears to be a (\w+) subject\."
    
    match = re.search(pattern, text)
    
    if match:
        return match.group(1)  # Return found word
    else:
        return None  # Return None if pattern not found




import json

# Set file path (file must exist at execution location)
input_file = '/Users/apple/Desktop/neuro-ai-research-system/projects/BrainVLM/code/BrainVLM-umbrella/UMBRELLA/predictions/sex_comparison_conversations_simple_extended_output_filtered.jsonl'

total = 0 
correct = 0 
wrong = 0 

subject_ids = [] 

print("Starting filtering...")


try:
    with open(input_file, 'r', encoding='utf-8') as f_in:
        subject_id = None
        for i, line in enumerate(f_in):
            line = line.strip()
            if not line: continue  # Skip empty lines

            try:
                data = json.loads(line)
                
                # Extract subject_ids list according to JSON structure
                # Structure: root -> metadata -> sample_metadata -> subject_ids
                
                # Safe key access (prevent KeyError)
                subject_id_tmp = data['metadata']['sample_metadata'].get('subject_id', None)
                label = data['metadata']['sample_metadata'].get('subject_label', [])
                generated_text = data.get('model_answer', [])
                pred = extract_label(generated_text)
                subject_ids.append(subject_id_tmp)
                if i == 0: 
                    subject_id = subject_id_tmp


                if subject_id_tmp == subject_id: 
                    pass 
                else: 
                    if label == pred: 
                        correct += 1 
                    else: 
                        print(f"SubjectID: {subject_id}, LABEL: {label}, PRED: {pred}")
                        wrong += 1 
                    total +=1
                    subject_id = subject_id_tmp
            
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON line: {line[:50]}...")

    print("-" * 30)
    print(f"Task completed!")
    print(f"Total: {total}, Correct: {correct}, Wrong: {wrong}, ACC: {(correct/total)}")
    import numpy as np 
    print(f"Total Subject: {len(np.unique(subject_ids,return_counts=True)[0])}")


except FileNotFoundError:
    print(f"Error: File '{input_file}' not found. Please check the file path.")