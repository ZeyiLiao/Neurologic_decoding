cd seq2seq


# t5-large
PYTHONPATH=.. python decode.py --model_name t5-large --input_path ../dataset/clean/t5_input.txt --reference_path\
 ../dataset/clean/commongen.1.tgt.txt --constraint_file ../dataset/clean/constraint/constraint_inflections.json --constraint_file_lemma ../dataset/clean/constraint/constraint_lemmas.json --min_tgt_length 5\
  --max_tgt_length 64 --bs 32 --beam_size 20 --length_penalty 0.1 --ngram_size 3 --prune_factor 50000 --sat_tolerance 2 --beta 1\
   --early_stop 1 --save_path ../output_dir/output_file_t5_large.1 --score_path ../output_dir/output_file_t5_large.json



# finetuned_ckpt and with mask in inputs --t5-3b
# note: since we trained t5-3b on V4 transformer and seems some layer of it don't match the layer at V3

PYTHONPATH=.. python decode.py --model_name /home/zeyi/transformers/ckpt/t5_3b_w_m/checkpoint-18 --input_path /home/zeyi/longtail/longtail_data/raw_data/inputs_t5_infer.csv --reference_path\
 ../dataset/clean/commongen.1.tgt.txt --constraint_file /home/zeyi/longtail/longtail_data/raw_data/inflection_constraints_t5_infer.json --constraint_file_lemma /home/zeyi/longtail/longtail_data/raw_data/lemma_constraints_t5_infer.json --min_tgt_length 5\
  --max_tgt_length 64 --bs 8 --beam_size 20 --length_penalty 0.1 --ngram_size 3 --prune_factor 50000 --sat_tolerance 2 --beta 1\
   --early_stop 1 --save_path /home/zeyi/longtail/longtail_data/generated_data/t5_3b_w_m.csv --score_path ../output_dir/output_file_t5_3b.json --task constrained_generation




# for dis
PYTHONPATH=.. python decode.py --model_name /home/zeyi/transformers/ckpt/t5_3b_w_m_dis/checkpoint-32 --input_path /home/zeyi/longtail/longtail_data/raw_data/for_dis/inputs_t5_infer.csv --reference_path\
 ../dataset/clean/commongen.1.tgt.txt --constraint_file /home/zeyi/longtail/longtail_data/raw_data/for_dis/inflection_constraints_t5_infer.json --constraint_file_lemma /home/zeyi/longtail/longtail_data/raw_data/for_dis/lemma_constraints_t5_infer.json --min_tgt_length 5\
  --max_tgt_length 64 --bs 8 --beam_size 20 --length_penalty 0.1 --ngram_size 3 --prune_factor 50000 --sat_tolerance 2 --beta 1\
   --early_stop 1 --save_path /home/zeyi/longtail/longtail_data/generated_data/for_dis/t5_3b_w_m.csv --score_path ../output_dir/output_file_t5_3b.json --task constrained_generation



# vanilla t5-3b with mask
PYTHONPATH=.. python run_eval.py \
  --model_name /home/zeyi/transformers/ckpt/t5_3b_w_m/checkpoint-18  \
  --input_path /home/zeyi/longtail/longtail_data/raw_data/inputs_t5_infer.csv --reference_path ../dataset/clean/commongen.1.tgt.txt  \
  --min_tgt_length 5 --max_tgt_length 64 --constraint_file_lemma /home/zeyi/longtail/longtail_data/raw_data/lemma_constraints_t5_infer.json \
  --bs 16 --beam_size 20 --length_penalty 0.1 --ngram_size 3 \
  --save_path /home/zeyi/longtail/longtail_data/generated_data/t5_3b_vanilla_w_m.csv --score_path ../output_dir/output_file_t5_3b.json


# for dis
PYTHONPATH=.. python run_eval.py \
  --model_name /home/zeyi/transformers/ckpt/t5_3b_w_m_dis/checkpoint-32  \
  --input_path /home/zeyi/longtail/longtail_data/raw_data/for_dis/inputs_t5_infer.csv --reference_path ../dataset/clean/commongen.1.tgt.txt  \
  --min_tgt_length 5 --max_tgt_length 64 --constraint_file_lemma /home/zeyi/longtail/longtail_data/raw_data/for_dis/lemma_constraints_t5_infer.json \
  --bs 16 --beam_size 20 --length_penalty 0.1 --ngram_size 3 \
  --save_path /home/zeyi/longtail/longtail_data/generated_data/for_dis/t5_3b_vanilla_w_m.csv --score_path ../output_dir/output_file_t5_3b.json
