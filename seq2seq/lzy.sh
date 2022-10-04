cd seq2seq


# t5-large
PYTHONPATH=.. python decode.py --model_name t5-large --input_path ../dataset/clean/t5_input.txt --reference_path\
 ../dataset/clean/commongen.1.tgt.txt --constraint_file ../dataset/clean/constraint/constraint_inflections.json --constraint_file_lemma ../dataset/clean/constraint/constraint_lemmas.json --min_tgt_length 5\
  --max_tgt_length 64 --bs 32 --beam_size 20 --length_penalty 0.1 --ngram_size 3 --prune_factor 50000 --sat_tolerance 2 --beta 1\
   --early_stop 1 --save_path ../output_dir/output_file_t5_large.1 --score_path ../output_dir/output_file_t5_large.json


# finetuned_ckpt  and with mask in inputs --t5-large
PYTHONPATH=.. python decode.py --model_name /home/zeyi/finetune/saved/lrgenerative_gpt_t5_large_w_m_t5_26_09_2022_f7dd9c2e/checkpoints/epoch=5-step=36.ckpt --input_path ../dataset/clean/t5_input_w_mask.txt --reference_path\
 ../dataset/clean/commongen.1.tgt.txt --constraint_file ../dataset/clean/constraint/constraint_inflections.json --constraint_file_lemma ../dataset/clean/constraint/constraint_lemmas.json --min_tgt_length 5\
  --max_tgt_length 64 --bs 32 --beam_size 20 --length_penalty 0.1 --ngram_size 3 --prune_factor 50000 --sat_tolerance 2 --beta 1\
   --early_stop 1 --save_path ../output_dir/t5_large_w_m.1 --score_path ../output_dir/output_file_t5_large.json --task constrained_generation --pl_ckpt --pl_model t5-large

# finetuned_ckpt  and without mask in inputs --t5-large
PYTHONPATH=.. python decode.py --model_name /home/zeyi/finetune/saved/lrgenerative_gpt_t5_large_wo_m_26_09_2022_d90dea10/checkpoints/epoch=5-step=36.ckpt --input_path ../dataset/clean/t5_input_wo_mask.txt --reference_path\
 ../dataset/clean/commongen.1.tgt.txt --constraint_file ../dataset/clean/constraint/constraint_inflections.json --constraint_file_lemma ../dataset/clean/constraint/constraint_lemmas.json --min_tgt_length 5\
  --max_tgt_length 64 --bs 32 --beam_size 20 --length_penalty 0.1 --ngram_size 3 --prune_factor 50000 --sat_tolerance 2 --beta 1\
   --early_stop 1 --save_path ../output_dir/t5_large_wo_m.1 --score_path ../output_dir/output_file_t5_large.json --task constrained_generation --pl_ckpt --pl_model t5-large



# finetuned_ckpt and with mask in inputs --t5-3b
# note: since we trained t5-3b on V4 transformer and seems some layer of it don't match the layer at V3
PYTHONPATH=.. python decode.py --model_name /home/zeyi/transformers/ckpt/t5_3b_w_m/checkpoint-12 --input_path ../dataset/clean/t5_input_w_mask.txt --reference_path\
 ../dataset/clean/commongen.1.tgt.txt --constraint_file ../dataset/clean/constraint/constraint_inflections.json --constraint_file_lemma ../dataset/clean/constraint/constraint_lemmas.json --min_tgt_length 5\
  --max_tgt_length 64 --bs 8 --beam_size 20 --length_penalty 0.1 --ngram_size 3 --prune_factor 50000 --sat_tolerance 2 --beta 1\
   --early_stop 1 --save_path ../output_dir/t5_3b_w_m.1 --score_path ../output_dir/output_file_t5_large.json --task constrained_generation


# finetuned_ckpt and without mask in inputs --t5-3b
PYTHONPATH=.. python decode.py --model_name /home/zeyi/transformers/ckpt/t5_3b_wo_m/checkpoint-12 --input_path ../dataset/clean/t5_input_wo_mask.txt --reference_path\
 ../dataset/clean/commongen.1.tgt.txt --constraint_file ../dataset/clean/constraint/constraint_inflections.json --constraint_file_lemma ../dataset/clean/constraint/constraint_lemmas.json --min_tgt_length 5\
  --max_tgt_length 64 --bs 8 --beam_size 20 --length_penalty 0.1 --ngram_size 3 --prune_factor 50000 --sat_tolerance 2 --beta 1\
   --early_stop 1 --save_path ../output_dir/t5_3b_wo_m.1 --score_path ../output_dir/output_file_t5_large.json --task constrained_generation





# vanilla t5-3b with mask
PYTHONPATH=.. python run_eval.py \
  --model_name /home/zeyi/transformers/ckpt/t5_3b_w_m/checkpoint-12  \
  --input_path ../dataset/clean/t5_input_w_mask.txt --reference_path ../dataset/clean/commongen.1.tgt.txt  \
  --min_tgt_length 5 --max_tgt_length 64 --constraint_file_lemma ../dataset/clean/constraint/constraint_lemmas.json \
  --bs 16 --beam_size 20 --length_penalty 0.1 --ngram_size 3 \
  --save_path ../output_dir/t5_3b_vanilla_w_m.1 --score_path ../output_dir/output_file_t5_large.json

# vanilla t5-3b w/o mask

PYTHONPATH=.. python run_eval.py \
  --model_name /home/zeyi/transformers/ckpt/t5_3b_wo_m/checkpoint-12  \
  --input_path ../dataset/clean/t5_input_wo_mask.txt --reference_path ../dataset/clean/commongen.1.tgt.txt  \
  --min_tgt_length 5 --max_tgt_length 64 --constraint_file_lemma ../dataset/clean/constraint/constraint_lemmas.json \
  --bs 8 --beam_size 20 --length_penalty 0.1 --ngram_size 3 \
  --save_path ../output_dir/t5_3b_vanilla_wo_m.1 --score_path ../output_dir/output_file_t5_large.json