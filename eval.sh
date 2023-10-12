# MODEL=<PATH_TO_MODEL>
# --bt_steps 'python_sa-cpp_sa-python_sa,cpp_sa-python_sa-cpp_sa'    \
python3 codegen_sources/model/train.py \
--eval_only true \
--exp_name transcoder_eval \
--dump_path '/home/alex/dump_codegen' \
--data_path '/home/alex/data/transcoder_test_set/test_dataset' \
--mt_steps 'python_sa-cpp_sa' \
--lgs 'python_sa-cpp_sa'  \
--encoder_only False \
--n_layers 0  \
--local_rank -1  \
--n_layers_encoder 6  \
--n_layers_decoder 6 \
--emb_dim 1024  \
--n_heads 8  \
--eval_bleu false \
--eval_bt_pairs false \
--eval_computation true \
--has_sentence_ids "valid|para,test|para" \
--generate_hypothesis false \
--save_periodic 1 \
--reload_model "$MODEL,$MODEL" \
--reload_encoder_for_decoder false \
--n_sentences_eval 1 \
--batch_size 1
# --use_goatswitch false \