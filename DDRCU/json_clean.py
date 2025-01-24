import json
import os
# 파일 경로 설정
input_file_path = './_reformat/train.json'  # 기존 파일 경로
output_file_path = './_reformat/GenConv.json'  # 저장할 파일 경로

# 기존 JSON 데이터 로드
if not os.path.exists(input_file_path):
    raise FileNotFoundError(f"{input_file_path} 파일이 존재하지 않습니다.")

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# "original_data" 필드를 바깥으로 빼고 삭제

def restructure_data(data):
    cleaned_data = []
    for entry in data:
        if isinstance(entry, dict) and 'original_data' in entry:
            original_data = entry.pop('original_data')  # "original_data" 필드 추출
            if isinstance(original_data, dict):
                entry.update(original_data)  # "original_data" 내용을 바깥으로 병합
        cleaned_data.append(entry)
    return cleaned_data

cleaned_data = restructure_data(data)

# 수정된 데이터를 새로운 파일에 저장
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"Restructured data saved to {output_file_path}")
