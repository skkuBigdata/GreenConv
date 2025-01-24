CUDA_VISIBLE_DEVICES=0 python prepare.py \
    --config_name strat \
    --inputter_name strat \
    --train_input_file ./DATA/train.txt \
    --max_input_length 512 \
    --max_decoder_input_length 50 \
    --use_all_persona False \
    --encode_context True