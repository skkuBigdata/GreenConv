import json

def json_to_txt(input_json_path, output_txt_path):
    with open(input_json_path, 'r', encoding='utf-8') as infile:
        try:
            data = json.load(infile) 
        except json.JSONDecodeError as e:
            print(f"JSON 파일 파싱 오류: {e}")
            return
    
    if not isinstance(data, list):
        print("JSON 데이터가 리스트 형식이어야 합니다.")
        return

    with open(output_txt_path, 'w', encoding='utf-8') as outfile:
        for item in data:
            json_str = json.dumps(item, ensure_ascii=False) 
            outfile.write(json_str + '\n')  
    
    print(f"JSON 데이터를 {output_txt_path}에 성공적으로 저장했습니다.")


input_json_path = './GenConv5.json'  
output_txt_path = './train.txt'  

json_to_txt(input_json_path, output_txt_path)