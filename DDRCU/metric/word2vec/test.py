from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# GloVe 텍스트 파일 경로 (원본 경로)
input_path = './glove.6B.300d.txt'

# 변환된 Word2Vec 텍스트 파일 경로
word2vec_path = './glove.6B.300d.word2vec.txt'

# 변환된 Word2Vec 바이너리 파일 경로
binary_path = './glove.6B.300d.model.bin'

# 1. GloVe 파일을 Word2Vec 포맷으로 변환
print("Step 1: Converting GloVe format to Word2Vec format...")
glove2word2vec(input_path, word2vec_path)
print(f"GloVe to Word2Vec conversion completed. Saved to: {word2vec_path}")

# 2. 변환된 Word2Vec 파일 로드
print("Step 2: Loading the Word2Vec format file...")
model = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
print(f"Loaded Word2Vec format file: {word2vec_path}")

# 3. 바이너리 파일로 저장
print("Step 3: Saving the model as a binary file...")
model.save_word2vec_format(binary_path, binary=True)
print(f"Binary file saved to: {binary_path}")
