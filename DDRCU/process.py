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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

random.seed(13)

relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def str2bool(x):
    if x == "True":
        return True
    elif x == "False":
        return False
    else:
        raise argparse.ArgumentTypeError("must be True or False")

parser = argparse.ArgumentParser()
parser.add_argument('--add_persona', type=str2bool, required=True,
                    help="True or False, 페르소나 데이터를 사용할지 여부 설정")
args = parser.parse_args()

def _norm(x):
    return ' '.join(x.strip().split())


def process_sent(sentence):
    sentence = sentence.lower()
    for k, v in word_pairs.items():  
        sentence = sentence.replace(k, v)
    sentence = nltk.word_tokenize(sentence) 
    return sentence

strategies = json.load(open('./_reformat/strategy.json'))
strategies = [e[1:-1] for e in strategies]
strat2id = {strat: i for i, strat in enumerate(strategies)} 
print(f"페르소나 사용 여부: {args.add_persona}")


def get_commonsense(comet, item):
    cs_list = []  
    input_event = " ".join(item)
    for rel in relations:
        try:
            cs_res = comet.generate(input_event, rel)
            print(cs_res)
            cs_list.append(cs_res)
        except Exception as e:
            print(f"COMET generation Error - Relation: '{rel}', input: '{input_event}': {e}")
            cs_list.append([]) 
    return cs_list

def comet_data(d):
    try:
        d = eval(d)  
        emotion = d['emotion_type']  
        problem = d["problem_type"] 
        situation = d['situation'] 
        persona = d['persona'] 
        persona_list = d['persona_list']  
        d = d['dialog']  
        dial = []  
        pre_text = ""  
        context = ""  
        user_number = 0 

        for uttr in d:
            text = _norm(uttr['text'])  
            role = uttr['speaker']  

            if role == 'usr': 
                dial.append({
                    'text': text,  
                    'speaker': 'usr', 
                })
                user_number += 1 
            else:  
                if len(uttr) > 3:  
                    persona = persona_list[user_number - 3]  

                    post = _norm(pre_text + persona)
                    item = process_sent(post)  
                    cs_list = get_commonsense(comet, item)  

                    dial.append({
                        'text': text,  
                        'speaker': 'sys',
                        'strategy': uttr['strategy'],  
                        'dpr': uttr['dpr'], 
                        'comet': cs_list  
                    })
                else:  
                    dial.append({
                        'text': text,
                        'speaker': 'sys',
                        'strategy': uttr['strategy'],
                    })
            pre_text = uttr['text'] 
            context += pre_text 

        res = {
            'emotion_type': emotion, 
            'problem_type': problem, 
            'persona': persona,  
            'persona_list': persona_list,  
            'situation': situation, 
            'dialog': dial,  
        }
        print(res)
        return res

    except Exception as e:
        print(f"Error processing data: {e}")
        return None

comet = Comet("/home/sj/DDRCU/src/utils/comet2", device)

datasets = ['test.txt', 'valid.txt', 'train.txt']  
for dataset in datasets:
    try:
        print(f"파일 {dataset} 처리 시작...")  
        with open('./_reformat/' + dataset) as f:
            original = f.readlines() 

        output_file = './DATA/' + dataset

        with open(output_file, 'w') as f_out:
            for e in tqdm.tqdm(original, total=len(original)):
                try:
                    processed_data = comet_data(e) 
                    if processed_data is not None:  
                        f_out.write(json.dumps(processed_data) + '\n')
                        
                        f_out.flush()  
                except Exception as e:
                    print(f"Error occured while processing data: {e}")

        print(f"file {output_file} saved!") 

    except Exception as e:
        print(f"Error occured while processing {dataset}: {e}")

try:
    for dataset in datasets:
        emotions = Counter([e['emotion_type'] for e in dataset])
        problems = Counter([e['problem_type'] for e in dataset])
except Exception as e:
    print(e)