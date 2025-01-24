# Copyright (C) 2016 Manas Ranjan Kar <manasrkar91@gmail.com>
# GNU LGPL v2.1 라이센스 - http://www.gnu.org/licenses/lgpl.html

"""
사용법:
    $ python -m gensim.scripts.glove2word2vec --input <GloVe 벡터 파일> --output <Word2Vec 벡터 파일>

형식:
GloVe 포맷 (예제는 `Stanford <https://nlp.stanford.edu/projects/glove/>`에서 확인 가능) ::
    word1 0.123 0.134 0.532 0.152
    word2 0.934 0.412 0.532 0.159
    word3 0.334 0.241 0.324 0.188
    ...
    word9 0.334 0.241 0.324 0.188
Word2Vec 포맷 (예제는 `w2v 레포지토리 <https://code.google.com/archive/p/word2vec/>`에서 확인 가능) ::
    9 4
    word1 0.123 0.134 0.532 0.152
    word2 0.934 0.412 0.532 0.159
    word3 0.334 0.241 0.324 0.188
    ...
    word9 0.334 0.241 0.324 0.188
사용 방법
----------
>>> from gensim.test.utils import datapath, get_tmpfile
>>> from gensim.models import KeyedVectors
>>>
>>> glove_file = datapath('test_glove.txt')
>>> tmp_file = get_tmpfile("test_word2vec.txt")
>>>
>>> # glove2word2vec 스크립트 호출
>>> # 기본 방식 (CLI 사용): python -m gensim.scripts.glove2word2vec --input <glove_file> --output <w2v_file>
>>> from gensim.scripts.glove2word2vec import glove2word2vec
>>> glove2word2vec(glove_file, tmp_file)
>>>
>>> model = KeyedVectors.load_word2vec_format(tmp_file)
커맨드 라인 인자
----------------------
.. 프로그램 출력 예: python -m gensim.scripts.glove2word2vec --help
   :ellipsis: 0, -5
"""
import sys
import logging
import argparse

from smart_open import smart_open
logger = logging.getLogger(__name__)


def get_glove_info(glove_file_name):
    """GloVe 포맷 파일에서 벡터 수와 차원을 반환합니다."""
    """입력된 `glove_file_name`의 벡터 수와 벡터 차원을 가져옵니다.
    매개변수
    ----------
    glove_file_name : str
        GloVe 포맷 파일 경로.
    반환값
    -------
    (int, int)
        입력 파일의 벡터 수 (라인 수)와 벡터 차원.
    """
    with smart_open(glove_file_name) as f:
        num_lines = sum(1 for _ in f)
    with smart_open(glove_file_name) as f:
        num_dims = len(f.readline().split()) - 1
    return num_lines, num_dims


def glove2word2vec(glove_input_file, word2vec_output_file):
    """`glove_input_file`의 GloVe 포맷을 Word2Vec 포맷의 `word2vec_output_file`로 변환합니다."""
    """GloVe 포맷의 `glove_input_file`을 Word2Vec 포맷으로 변환하여 `word2vec_output_file`에 저장합니다.
    매개변수
    ----------
    glove_input_file : str
        GloVe 포맷 파일 경로.
    word2vec_output_file: str
        출력 파일 경로.
    반환값
    -------
    (int, int)
        입력 파일의 벡터 수 (라인 수)와 벡터 차원.
    """
    num_lines, num_dims = get_glove_info(glove_input_file)
    logger.info("총 %i개의 벡터를 %s에서 %s로 변환 중입니다", num_lines, glove_input_file, word2vec_output_file)
    with smart_open(word2vec_output_file, 'wb') as fout:
        fout.write("{0} {1}\n".format(num_lines, num_dims).encode('utf-8'))
        with smart_open(glove_input_file, 'rb') as fin:
            for line in fin:
                fout.write(line)
    return num_lines, num_dims


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(level=logging.INFO)
    logger.info("실행 중 %s", ' '.join(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="GloVe 포맷의 입력 파일 (읽기 전용).")
    parser.add_argument(
        "-o", "--output", required=True, help="Word2Vec 텍스트 포맷의 출력 파일 (덮어쓰기)."
    )
    logging.basicConfig(format='%(asctime)s - %(module)s - %(levelname)s - %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__[:-135], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="GloVe 포맷의 입력 파일 경로")
    parser.add_argument("-o", "--output", required=True, help="출력 파일 경로")
    args = parser.parse_args()

    # 실제 변환 실행
    logger.info("실행 중 %s", ' '.join(sys.argv))
    num_lines, num_dims = glove2word2vec(args.input, args.output)
    logger.info('변환 완료: %i개의 벡터와 %i개의 차원', num_lines, num_dims)
