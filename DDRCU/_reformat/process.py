import json
import pickle
import tqdm
import numpy as np
import multiprocessing as mp
import nltk
import random
from collections import Counter
import argparse

random.seed(13)

# 문자열을 불리언으로 변환하는 함수
def str2bool(x):
    if x == "True":
        return True
    elif x == "False":
        return False
    else:
        raise argparse.ArgumentTypeError("must be True or False")

# 명령줄 인자 설정
parser = argparse.ArgumentParser()
parser.add_argument('--add_persona', type=str2bool, required=True,
                    help="True or False, this determine whether to use ESConv with persona")
args = parser.parse_args()

# 문자열 정규화 함수
def _norm(x):
    return ' '.join(x.strip().split())

# 전략 로드 및 전처리
strategies = json.load(open('./strategy.json'))
strategies = [e[1:-1] for e in strategies]
strat2id = {strat: i for i, strat in enumerate(strategies)}

print(f"Processing with add_persona={args.add_persona}")
if args.add_persona:
    original = json.load(open('./DPRConv_6.json'))
else:
    original = json.load(open('./ESConv.json'))

# 데이터 처리 함수
def process_data(diglog):
    # 데이터에서 감정 유형, 문제 유형, 상황, 페르소나 정보를 추출
    emotion = diglog['emotion_type']  # 감정 유형
    problem = diglog["problem_type"]  # 문제 유형
    situation = diglog['situation']   # 상황
    persona = diglog['persona']       # 페르소나 설명 (단일 텍스트)
    persona_list = diglog['persona_list']  # 페르소나의 세부 정보 (리스트 형태)

    # 대화 데이터를 순회하며 사용자와 시스템 발화를 처리
    diglog = diglog['dialog']
    dialList = []  # 처리된 대화 데이터 저장을 위한 리스트
    for uttr in diglog:
        # 발화 내용 정규화 (공백 제거 및 단어 정리)
        text = _norm(uttr['content'])
        role = uttr['speaker']  # 발화자 역할 (사용자: 'seeker', 시스템: 'sys')
        if role == 'seeker':  # 발화자가 사용자일 경우
            dialList.append({
                'text': text,  # 발화 내용
                'speaker': 'usr',  # 발화자 태그를 'usr'로 설정
            })
        else:  # 발화자가 시스템일 경우
            # 시스템 발화에 추가 정보(annotation)가 포함된 경우
            # len(uttr)은 uttr 딕셔너리에 포함된 키의 개수
            if 'annotation' in uttr and 'dpr' in uttr:
                # 추가 정보가 있는 경우
                dialList.append({
                    'text': text,
                    'speaker': 'sys',
                    'strategy': uttr['annotation']['strategy'],
                    'dpr': uttr['dpr']
                })
            else:
                # 기본 정보만 있는 경우
                dialList.append({
                    'text': text,
                    'speaker': 'sys',
                    'strategy': uttr['annotation']['strategy']
                })


    # 최종 처리된 데이터 구조 반환
    return {
        'emotion_type': emotion,  # 감정 유형
        'problem_type': problem,  # 문제 유형
        'persona': persona,  # 페르소나 설명
        'persona_list': persona_list,  # 페르소나 세부 정보
        'situation': situation,  # 상황 설명
        'dialog': dialList,  # 처리된 대화 데이터
    }


# 데이터 처리
data = []
with mp.Pool(processes=mp.cpu_count()) as pool:
    for e in pool.imap(process_data, tqdm.tqdm(original, total=len(original))):
        data.append(e)

# 데이터 통계
emotions = Counter([e['emotion_type'] for e in data])
problems = Counter([e['problem_type'] for e in data])
print('emotion:', emotions)
print('problem:', problems)

# 데이터 셔플 및 분할
random.shuffle(data)
dev_size = int(0.1 * len(data))
test_size = int(0.1 * len(data))
valid = data[:dev_size] + data[dev_size + test_size: dev_size + dev_size + test_size]
test = data[dev_size: dev_size + test_size]
train = data[dev_size + dev_size + test_size:]

# 파일 저장 함수
def save_data(filename, dataset):
    print(f"Saving data to {filename}...")
    try:
        with open(filename, 'w') as f:
            for e in dataset:
                f.write(json.dumps(e, ensure_ascii=False) + '\n')
        print(f"{filename} saved successfully!")
        # 저장된 파일 내용 일부 출력
        with open(filename, 'r') as f:
            print(f"Contents of {filename} (first 3 lines):")
            for i, line in enumerate(f):
                print(line.strip())
                if i == 2:  # 첫 3줄만 출력
                    break
    except Exception as e:
        print(f"Error saving {filename}: {e}")

# 데이터 저장 및 처리 통계
print(f"Processing train set: {len(train)} samples")
save_data('train.txt', train)

print(f"Processing valid set: {len(valid)} samples")
save_data('valid.txt', valid)

print(f"Processing test set: {len(test)} samples")
save_data('test.txt', test)

# 전체 평균 대화 길이 및 발화 길이 계산
turns = sum(len(e['dialog']) for e in train + valid + test)
uttrs = sum(len(uttr['text'].split()) for e in train + valid + test for uttr in e['dialog'])
print('Avg. length of dialogues:', turns / (len(train) + len(valid) + len(test)))
print('Avg. length of utterances:', uttrs / turns)
