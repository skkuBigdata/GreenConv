# ESCoT: 해석 가능한 감정 지원 대화 시스템을 향하여

<img src="https://img.shields.io/badge/Venue-ACL--24-278ea5" alt="venue"/> <img src="https://img.shields.io/badge/Status-Accepted-success" alt="status"/>

이 저장소는 ACL 2024 메인 논문 "[**ESCoT: Towards Interpretable Emotional Support Dialogue Systems**](https://aclanthology.org/2024.acl-long.723/)"의 소스입니다.

## ESD-CoT 데이터셋

우리의 ESD-CoT 데이터셋은 `data` 폴더에 있으며, `train`, `val`, `test`의 세 개 JSON 파일로 구성되어 있습니다. 각 파일은 다음과 같은 구조의 샘플을 포함합니다:

```json
{
    "id": ,
    "original_data": {
        "dialog": [
            {
                "speaker": "seeker",
                "content": "Hi, I'm having a really hard time managing my schoolwork and extracurricular activities. I feel like there's just not enough hours in the day."
            },
            ...
            {
                "speaker": "seeker",
                "content": "Yeah, I can try that."
            }
        ],
        "strategy": "Providing Suggestions",
        "response": "Great, and let's touch base next week to see if the list has been helpful. In the meantime, have you considered talking to your teacher or a guidance counselor about feeling overwhelmed?"
    },
    "cot_data": {
        "emotion": "The seeker feels overwhelmed and stretched thin.",
        "emotion_stimuli": "The seeker is struggling to manage schoolwork...",
        "individual_appraisal": "The seeker thinks they are not able to do anything well...",
        "recognized_strategy": "Providing Suggestions",
        "strategy_reason": "To address the seeker's feeling of being overwhelmed and..."
    }
}
```

추가적으로, `data/ablation_data` 폴더에 instructional format으로 작성된 훈련 데이터를 제공합니다.

## 모델 학습

### Pretrained 모델 다운로드
[**LLAMA2-7B-CHAT**](https://huggingface.co/meta-llama/Llama-2-7b-hf) 모델을 다운로드합니다.

LLAMA2-CHAT 모델의 학습은 [**Transformer Reinforcement Learning의 SFT Trainer**](https://github.com/huggingface/trl)를 기반으로 합니다.

### 모델 학습 방법
- `scripts/supervised_finetune_llama2_cot.sh`를 실행하여 모델을 학습시킵니다.  
- Ablation Study 모델 학습을 위해 `scripts/supervised_finetune_llama2_cot_ablation.sh`를 실행합니다.

### 모델 테스트 방법
- `scripts/test_llama2_chat_sft_cot.sh` 또는 `scripts/test_llama2_inference_cot.sh`를 실행합니다.

## 논문 인용
연구가 본 논문과 관련이 있다면 아래 형식으로 논문을 인용해주세요:
```bib
@inproceedings{zhang-etal-2024-escot,
    title = "{ESC}o{T}: Towards Interpretable Emotional Support Dialogue Systems",
    author = "Zhang, Tenggan and Zhang, Xinjie and Zhao, Jinming and Zhou, Li and Jin, Qin",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2024"
}
```

문의 사항이 있을 경우 zhangxinjie827@ruc.edu.cn로 연락주시기 바랍니다.