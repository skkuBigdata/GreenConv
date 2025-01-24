import json
import tqdm
import multiprocessing as mp
import random
import torch
from collections import Counter
import os
random.seed(13)




problem_list = [
    "individual discomfort issues",
    "career development issues",
    "economic issues",
    "self-growth issues",
    "mental health issues",
    "social environment issues",
    "marriage and family issues",
    "interpersonal relationship issues",
    "Others"
]

texts_usr = {
    "individual discomfort issues": [],
    "career development issues": [],
    "economic issues": [],
    "self-growth issues": [],
    "mental health issues": [],
    "social environment issues": [],
    "marriage and family issues": [],
    "interpersonal relationship issues": [],
    "Others": []
}

texts_sys = {
    "individual discomfort issues": [],
    "career development issues": [],
    "economic issues": [],
    "self-growth issues": [],
    "mental health issues": [],
    "social environment issues": [],
    "marriage and family issues": [],
    "interpersonal relationship issues": [],
    "Others": []
}


texts_sys_strategy = {
    "individual discomfort issues": [],
    "career development issues": [],
    "economic issues": [],
    "self-growth issues": [],
    "mental health issues": [],
    "social environment issues": [],
    "marriage and family issues": [],
    "interpersonal relationship issues": [],
    "Others": []
}



strategy = {
    'toxic relationship': [],
    'friend betrayed': [],
    'breakup': [],
    'tension': [],
    'trouble communicating': [],
    'abusive relationship': [],
    'romantic relationship': [],
    'difficult roommate': [],
    'social isolation': [],
    'Others': []
}



original = json.load(open('./DPRConv_6.json'))



usr_list = []
sys_list = []
str_list = []
sys_list_str = []


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('train.txt') as f:
    reader = f.readlines()
for data in reader:
    data = eval(data)
    dialog = data['dialog']
    # persona_list = data['persona_list']
    persona = ''
    user_number = 0
    problem = data['problem_type']
    if problem not in problem_list:
        problem = 'Others'

    for i in range(len(dialog)):
        if dialog[i]['speaker'] != 'sys':
            user_number += 1

        if i > 0 and dialog[i]['speaker'] == 'sys':
            texts_usr[problem].append(dialog[i-1]['text'])
            texts_sys[problem].append(dialog[i]['text'])
            texts_sys_strategy[problem].append(persona + '[' + dialog[i]['strategy'] + '] ' + dialog[i]['text'])
            strategy[problem].append(dialog[i]['strategy'])


def _norm(x):
    return ' '.join(x.strip().split())


def add_dpr(data):
    dialog = data['dialog']
    persona_list = data['persona_list']
    persona = ""
    user_number = 0
    problem = data['problem_type']
    if problem not in problem_list:
        problem = 'others'

    for i in range(len(dialog)):
        if dialog[i]['speaker'] != 'sys':
            user_number += 1

        if i > 0 and dialog[i]['speaker'] == 'sys':
            last_text = _norm(dialog[i - 1]['content'])

            if dialog[i - 1]['speaker'] != 'sys' and user_number > 2:
                persona = persona_list[user_number - 3]
                # persona = persona.replace('<persona> ', '').replace(' <input>', '')

            with torch.no_grad():
                # 细节还需把控???
                # Internet search???
                encoded_inputs = DPR_tokenizer(
                    questions=[persona + last_text] * len(strategy[problem]),
                    # questions=[last_text] * len(strategy[problem]),
                    # titles=[dialog[i]['annotation']['strategy']] * len(strategy[problem]),
                    texts=texts_sys_strategy[problem],
                    return_tensors="pt",
                    padding='max_length',
                    truncation=True,
                    max_length=32
                ).to(device)
                outputs = DPR_reader(**encoded_inputs)
                relevance_logits = outputs.relevance_logits
                _, indices = torch.topk(relevance_logits, 10)
                dialog[i]['dpr'] = [[texts_usr[problem][index]] + [strategy[problem][index]] + [texts_sys[problem][index]] for index in indices]
                print(dialog[i]['dpr'])
                # dialog[i]['dpr'] = [[usr_list[index]] + [str_list[index]] + [sys_list[index]] for index in indices]

    data['dialog'] = dialog

    return data


#from transformers import DPRReaderTokenizerFast, DPRReader

#DPR_tokenizer = DPRReaderTokenizerFast.from_pretrained('../DPR-reader')
#DPR_reader = DPRReader.from_pretrained('../DPR-reader').to(device)

from transformers import T5Tokenizer, T5ForConditionalGeneration

DPR_tokenizer = T5Tokenizer.from_pretrained('../GenRet')
DPR_reader = T5ForConditionalGeneration.from_pretrained('../GenRet').to(device)

data = []
for d in tqdm.tqdm(original, total=len(original)):
    data.append(add_dpr(d))

with open('DPRConv_6.json', 'w', encoding='utf-8', errors='ignore') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)