# import json

# def generate_persona_and_list(sentences):
#     # persona 필드 생성
#     persona = ' '.join([f"<persona> {s}." for s in sentences]) + " <input>"
    
#     # persona_list 생성
#     persona_list = []
#     for i in range(1, len(sentences) + 1):
#         partial_persona = ' '.join([f"<persona> {s}." for s in sentences[:i]]) + " <input>"
#         persona_list.append(partial_persona)
    
#     return persona, persona_list

# def generate_persona_fields(input_path, output_path):
#     # JSON 파일 읽기
#     with open(input_path, 'r', encoding='utf-8') as infile:
#         data = json.load(infile)
    
#     for entry in data:
#         # 문장 분리
#         stimuli = entry['cot_data']['emotion_stimuli']
#         appraisal = entry['cot_data']['individual_appraisal']
#         sentences = stimuli.split('.') + appraisal.split('.')
#         sentences = [s.strip() for s in sentences if s.strip()]  # 빈 문장 제거

#         # persona와 persona_list 생성
#         persona, persona_list = generate_persona_and_list(sentences)
#         entry['persona'] = persona
#         entry['persona_list'] = persona_list
    
#     # 결과 저장
#     with open(output_path, 'w', encoding='utf-8') as outfile:
#         json.dump(data, outfile, indent=4, ensure_ascii=False)

# # 경로 설정
# input_path = './_reformat/GenConv3.json'
# output_path = './_reformat/GenConv4.json'

# # 함수 실행
# generate_persona_fields(input_path, output_path)



import json

def json_to_txt(input_json_path, output_txt_path):
    # JSON 파일 읽기
    with open(input_json_path, 'r', encoding='utf-8') as infile:
        try:
            data = json.load(infile)  # JSON 객체로 변환
        except json.JSONDecodeError as e:
            print(f"JSON 파일 파싱 오류: {e}")
            return
    
    # 데이터가 리스트인지 확인
    if not isinstance(data, list):
        print("JSON 데이터가 리스트 형식이어야 합니다.")
        return

    # 각 데이터를 한 줄로 저장
    with open(output_txt_path, 'w', encoding='utf-8') as outfile:
        for item in data:
            json_str = json.dumps(item, ensure_ascii=False)  # 데이터를 한 줄로 변환
            outfile.write(json_str + '\n')  # 한 줄씩 저장
    
    print(f"JSON 데이터를 {output_txt_path}에 성공적으로 저장했습니다.")


input_json_path = './GenConv5.json'  # JSON 파일 경로
output_txt_path = './train.txt'  # 변환된 텍스트 파일 경로

# 함수 실행
json_to_txt(input_json_path, output_txt_path)




# import json
# import re
# from collections import Counter
# from difflib import SequenceMatcher
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize

# # 전처리용 스템머
# stemmer = PorterStemmer()

# def preprocess_text(text):
#     """소문자 변환 및 불필요한 공백 제거"""
#     return text.lower().strip()

# def stem_keywords(keywords):
#     """키워드 리스트를 스템화"""
#     return [stemmer.stem(word) for word in keywords]

# def tokenize_and_stem(text):
#     """텍스트를 토큰화하고 스템화"""
#     words = word_tokenize(text)
#     return [stemmer.stem(word) for word in words]

# def map_problem(data, problem_mapping, threshold=0.6):
#     """카테고리 매핑"""
#     emotion_stimuli = preprocess_text(data['persona'])
#     tokenized_stimuli = tokenize_and_stem(emotion_stimuli)

#     # 빈도 기반 매칭
#     category_scores = Counter()
#     for category, stimuli_list in problem_mapping.items():
#         stemmed_stimuli_list = stem_keywords(stimuli_list)
#         for token in tokenized_stimuli:
#             if token in stemmed_stimuli_list:
#                 category_scores[category] += 1

#     # 유사도 기반 보완
#     for category, stimuli_list in problem_mapping.items():
#         for stimulus in stimuli_list:
#             similarity = SequenceMatcher(None, stimulus, emotion_stimuli).ratio()
#             if similarity >= threshold:
#                 category_scores[category] += similarity

#     # 가장 높은 점수를 가진 카테고리 반환
#     if category_scores:
#         return category_scores.most_common(1)[0][0]
#     return "Others"

# def add_problem_field(input_path, output_path, problem_mapping):
#     """JSON 데이터에 문제 카테고리 필드 추가"""
#     with open(input_path, 'r', encoding='utf-8') as infile:
#         data = json.load(infile)

#     for entry in data:
#         if 'persona' in entry:  # persona 필드가 있는지 확인
#             category = map_problem(entry, problem_mapping)
#             entry['problem_category'] = category  # 결과 저장
    
#     with open(output_path, 'w', encoding='utf-8') as outfile:
#         json.dump(data, outfile, indent=4, ensure_ascii=False)
#     print(f"결과가 {output_path}에 저장되었습니다.")



# # 문제 카테고리 매핑
# problem_mapping = {
    
    
#     "individual discomfort issues": [
#         "chronic illness", "disability", "burnout", "infertility", "strain", "injury", "illness",
#     "disorder", "chronic pain", "pandemic", "long-term illness", 
#         "physical limitations", "health challenges", "exhaustion", "fatigue", 
#         "emotional burnout", "physical strain", "gender dysphoria", "prolonged pain", 
#         "terminal illness", "health struggles", "sickness", "being tired", "not feeling well", 
#         "body aches", "ill health", "feeling weak", "constant pain", "ongoing health issues"
#     ],
#     "career development issues": [ "current career", "career", "job", "employment", 
#         "job search", "career prospect", "limited resources", "academic",
#         "workplace harassment", "career change", "difficult boss", "colleague",
#         "job loss", "heavy workload", "difficult coworker", "choosing a career", "schoolwork",
#         "lack of opportunities", "promotion struggles", "professional stagnation", 
#         "work-life balance", "work stress", "toxic workplace", "layoffs", "internship difficulties", 
#         "job insecurity", "team conflicts", "performance pressure", "finding a job", "bad manager", 
#         "too much work", "hard tasks", "work problems", "not liking my job", "changing jobs"
#     ],
#     "economic issues": [
#         "budget", "bills", "mounting debt", "financial difficulties", "money", "homeless",
#         "bankruptcy", "poverty", "loan repayment", "credit card debt", "mortgage stress", 
#         "income instability", "economic hardship", "financial insecurity", "cost of living", 
#         "student loans", "unemployment", "unexpected expenses", "savings depletion", "financial strain",
#         "low income", "paying rent", "not enough money", "struggling with bills", "being broke"
#     ],
#     "self-growth issues": [
#         "gender identity", "graduate", "body image", "life change", 
#         "life transition", "future prospect", "sexual orientation", 
#         "directionless", "feeling unfulfilled", "personal growth", "self-discovery", 
#         "identity crisis", "finding purpose", "self-esteem", "overcoming self-doubt", 
#         "aspirations", "goal setting", "developing confidence", "life goals", "self-worth",
#         "learning about myself", "feeling lost", "improving myself", "finding who I am", "life dreams"
#     ],
#     "mental health issues": [ "trauma", "traumatic",
#         "mental illness", "social anxiety", "depression", "traumatic events", "mental health", 
#         "emotional distress", "after-math", "experiencing loneliness", 
#         "panic attacks", "ptsd", "addiction", "car accident", "mood disorders", 
#         "coping mechanisms", "stress", "grief", "emotional overwhelm", "psychiatric conditions", 
#         "suicidal thoughts", "self-harm", "bipolar disorder", "eating disorders", "anxiety attacks",
#         "feeling sad", "nervousness", "being scared", "emotional pain", "bad mental state", 
#     ],
#     "social environment issues": [
#         "discrimination", "harassment", "abuse", "prejudice", "bullying", "feel out of place",
#         "immigrants", "culture shock", "assault", "inequality", "new city", 
#         "racism", "sexism", "xenophobia", "hate crimes", "social marginalization", 
#         "cultural adjustment", "language barriers", "exclusion", "ethnic bias", 
#         "religious intolerance", "social injustice", "systemic oppression", "integration struggles",
#         "feeling different", "not fitting in", "being new", "adjusting to a new place", "being judged"
#     ],
#     "marriage and family issues": [
#         "family conflict", "elderly parent", "caregiver", "loved"
#         "childcare responsibilities", "divorce", "single parent",
#         "domestic violence", "partner cheated", "newborn baby", "family", 
#         "relationship tension", "custody battles", "estranged family", 
#         "parenting challenges", "marital problems", "spousal arguments", "co-parenting", 
#         "blended families", "aging parents", "child support issues", "family estrangement",
#         "arguments at home", "problems with kids", "taking care of parents", "problems with partner", "family fights"
#     ],
#     "interpersonal relationship issues": [
#         "toxic relationship", "friend betrayed", "breakup", "tension", 
#         "trouble communicating", "abusive relationship", "partner", "close friends",
#         "romantic relationship", "difficult roommate", "social isolation", 
#         "friendship struggles", "dating problems", "commitment issues", "jealousy", 
#         "trust issues", "partner conflicts", "emotional distance", "clingy partner", 
#         "falling out", "relationship insecurities", "misunderstandings", "codependency",
#         "losing friends", "not making friends", "problems with roommate", "issues with partner", "feeling lonely"
#     ],
# }


# # 경로 설정
# input_path = './GenConv4.json'
# output_path = './GenConv5.json'

# # 함수 실행
# add_problem_field(input_path, output_path, problem_mapping)
