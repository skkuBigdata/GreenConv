import json
import pickle
import tqdm
import numpy as np
import multiprocessing as mp
import nltk
import random
from collections import Counter
import argparse
from src.utils.constants import WORD_PAIRS as word_pairs
from src.utils.comet import Comet
import torch
import os

# CUDA 디바이스 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 랜덤 시드 설정 (결과 재현 가능성을 위해)
random.seed(13)

# COMET 관계 정의 (COMET에서 추출할 관계 유형들)
# intent: 의도, need: 필요, want: 원하는 것, effect: 영향, react: 반응
relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]

# PyTorch 디바이스 설정 (CUDA 사용 가능 여부 확인)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 문자열을 불리언 값으로 변환하는 함수
def str2bool(x):
    if x == "True":
        return True
    elif x == "False":
        return False
    else:
        raise argparse.ArgumentTypeError("must be True or False")


# 명령줄 인자 파싱 설정
parser = argparse.ArgumentParser()
parser.add_argument('--add_persona', type=str2bool, required=True,
                    help="True or False, 페르소나 데이터를 사용할지 여부 설정")
args = parser.parse_args()


# 문자열 정규화 함수 (공백 제거 및 정리)
def _norm(x):
    return ' '.join(x.strip().split())


# 문장을 처리하여 소문자로 변환하고, 사전에 정의된 단어 치환 및 토크나이즈를 수행
def process_sent(sentence):
    sentence = sentence.lower()  # 소문자 변환
    for k, v in word_pairs.items():  # 사전에 정의된 단어 치환
        sentence = sentence.replace(k, v)
    sentence = nltk.word_tokenize(sentence)  # 토크나이즈
    return sentence


# 전략 정보를 JSON 파일에서 로드
strategies = json.load(open('./_reformat/strategy.json'))
strategies = [e[1:-1] for e in strategies]  # 문자열 처리
strat2id = {strat: i for i, strat in enumerate(strategies)}  # 전략별 ID 생성
print(f"페르소나 사용 여부: {args.add_persona}")


# COMET 필드 채우기 함수
def get_commonsense(comet, item):
    cs_list = []  # 상식 데이터를 저장할 리스트
    input_event = " ".join(item)  # 입력 이벤트를 문자열로 변환
    for rel in relations:
        try:
            cs_res = comet.generate(input_event, rel)
            print(cs_res)
            cs_list.append(cs_res)
        except Exception as e:
            # 예외 발생 시 빈 리스트를 추가하고 오류 메시지를 출력
            print(f"COMET 상식 생성 오류 - 관계: '{rel}', 입력: '{input_event}': {e}")
            cs_list.append([])  # 오류 발생 시 빈 데이터로 처리
    return cs_list



# 데이터를 처리하여 COMET 정보를 추가. txt 작성함수에 return
def comet_data(d):
    try:
        d = eval(d)  # 문자열 데이터를 딕셔너리로 변환
        emotion = d['emotion_type']  # 감정 유형 추출
        problem = d["problem_type"]  # 문제 유형 추출
        situation = d['situation']  # 상황 정보 추출
        persona = d['persona']  # 페르소나 정보 추출
        persona_list = d['persona_list']  # 페르소나 리스트 추출
        d = d['dialog']  # 대화 데이터 추출
        dial = []  # 처리된 대화를 저장할 리스트
        pre_text = ""  # 이전 발화 텍스트 저장
        context = ""  # 전체 컨텍스트 저장
        user_number = 0  # 사용자 발화 번호

        for uttr in d:
            text = _norm(uttr['text'])  # 발화 텍스트 정규화
            role = uttr['speaker']  # 발화자의 역할 ('usr' 또는 'sys')

            if role == 'usr':  # 사용자의 발화일 경우
                dial.append({
                    'text': text,  # 텍스트 저장
                    'speaker': 'usr',  # 발화자 역할 저장
                })
                user_number += 1  # 사용자 발화 카운트 증가
            else:  # 시스템의 발화일 경우
                if len(uttr) > 3:  # 시스템 발화에 추가 정보가 있는 경우
                    persona = persona_list[user_number - 3]  # 페르소나 데이터 추출

                    # 이전 발화(pre_text)와 페르소나를 합쳐 상식 데이터를 생성
                    post = _norm(pre_text + persona)
                    item = process_sent(post)  # 문장 처리
                    cs_list = get_commonsense(comet, item)  # COMET을 사용하여 상식 생성

                    dial.append({
                        'text': text,  # 발화 텍스트 저장
                        'speaker': 'sys',  # 발화자 역할 저장
                        'strategy': uttr['strategy'],  # 전략 정보 저장
                        'dpr': uttr['dpr'],  # DPR 정보 저장
                        'comet': cs_list  # 생성된 COMET 데이터 저장
                    })
                else:  # 시스템 발화에 추가 정보가 없는 경우
                    dial.append({
                        'text': text,
                        'speaker': 'sys',
                        'strategy': uttr['strategy'],
                    })
            pre_text = uttr['text']  # 이전 발화 업데이트
            context += pre_text  # 전체 컨텍스트에 추가

        res = {
            'emotion_type': emotion,  # 감정 유형
            'problem_type': problem,  # 문제 유형
            'persona': persona,  # 페르소나
            'persona_list': persona_list,  # 페르소나 리스트
            'situation': situation,  # 상황
            'dialog': dial,  # 처리된 대화 데이터
        }
        print(res)
        return res

    except Exception as e:
        # 데이터 처리 중 예외 발생 시 오류 메시지 출력
        print(f"데이터 처리 오류: {e}")
        return None


# COMET 모델 초기화
comet = Comet("/home/sj/DDRCU/src/utils/comet2", device)


# 데이터를 처리하고 파일에 저장
datasets = ['test.txt', 'valid.txt', 'train.txt']  # 처리할 데이터셋 파일 목록
for dataset in datasets:
    try:
        print(f"파일 {dataset} 처리 시작...")  # 파일 처리 시작 알림
        with open('./_reformat/' + dataset) as f:
            original = f.readlines()  # 파일 읽기

        output_file = './DATA/' + dataset

        # 처리된 데이터를 하나씩 기록
        with open(output_file, 'w') as f_out:
            for e in tqdm.tqdm(original, total=len(original)):
                try:
                    processed_data = comet_data(e)  # 데이터 처리
                    if processed_data is not None:  # 유효한 데이터만 기록
                        f_out.write(json.dumps(processed_data) + '\n')
                        
                        f_out.flush()  # 데이터 기록 즉시 디스크에 저장
                except Exception as e:
                    print(f"데이터 처리 중 오류 발생: {e}")

        print(f"파일 {output_file} 저장 완료!")  # 파일 저장 완료 알림

    except Exception as e:
        # 파일 처리 중 예외 발생 시 오류 메시지 출력
        print(f"파일 {dataset} 처리 중 오류 발생: {e}")

# 전체 감정 및 문제 유형 통계 출력
try:
    for dataset in datasets:
        emotions = Counter([e['emotion_type'] for e in dataset])
        problems = Counter([e['problem_type'] for e in dataset])
        print('감정 통계:', emotions)
        print('문제 유형 통계:', problems)
except Exception as e:
    print(e)