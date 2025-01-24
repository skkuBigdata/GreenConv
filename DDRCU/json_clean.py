import json
import os
input_file_path = './_reformat/train.json' 
output_file_path = './_reformat/GenConv.json' 

if not os.path.exists(input_file_path):
    raise FileNotFoundError(f"{input_file_path} 파일이 존재하지 않습니다.")

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

def restructure_data(data):
    cleaned_data = []
    for entry in data:
        if isinstance(entry, dict) and 'original_data' in entry:
            original_data = entry.pop('original_data')  
            if isinstance(original_data, dict):
                entry.update(original_data) 
        cleaned_data.append(entry)
    return cleaned_data

cleaned_data = restructure_data(data)

with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"Restructured data saved to {output_file_path}")
