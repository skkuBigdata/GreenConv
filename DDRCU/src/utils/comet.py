import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sys
sys.path.append('/home/sj/DDRCU/src')

from utils.comet2.cutils import use_task_specific_params, trim_batch
from typing import List


class Comet:
    def __init__(self, model_path: str, device):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)
        self.batch_size = 1
        self.decoder_start_token_id = None

    def generate(self, input_event: str, rel: str, num_generate: int = 1) -> List[str]:
        query = f"{input_event} {rel} [GEN]"  # 쿼리 생성

        with torch.no_grad():
            # 입력 데이터 토크나이즈
            batch = self.tokenizer([query], return_tensors="pt", truncation=True, padding="max_length").to(self.device)
            input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

            # 텍스트 생성
            summaries = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_start_token_id=self.decoder_start_token_id,
                num_beams=1,  # Beam Search 활성화
                num_return_sequences=num_generate,
            )


            # 결과 디코딩
            results = self.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return results
