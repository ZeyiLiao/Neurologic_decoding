from cgi import test
from distutils.command.config import config
from operator import mod
import os
import argparse
import json
from pathlib import Path
import readline
from tkinter.messagebox import NO
from unittest import result
from collections import OrderedDict
import types

import torch
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,T5ForConditionalGeneration,AutoConfig

from finetuned_model.lrgenerative import LRGenerative

from seq2seq.utils import calculate_rouge, use_task_specific_params, calculate_bleu_score, trim_batch
from unilm import utils_seq2seq
from lexical_constraints import init_batch

from seq2seq.generate import generate

#from zero_shot.generate_baseline import generate

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def filter(inputs, outputs, end_token='<extra_id_1>'):
    results = []
    for index in range(len(inputs)):
        input = inputs[index]
        output = outputs[index]

        _0_index = input.index('<extra_id_0>')
        _result_prefix = input[:_0_index][input[:_0_index].rfind('Input: ')+7:]
        # 12 is the length of <extra_id_0>
        # _result_suffix = input[_0_index+12:-5]
        _result_suffix = input[_0_index:][12:input[_0_index:].find(';')]

        # The first token is <unk> (inidex at 0) and the second token is <extra_id_0> (indexed at 32099)
        if end_token in output:

            _end_token_index = output.index(end_token)
            results.append((_result_prefix + output[12:_end_token_index] + _result_suffix).replace('  ',' '))
        else:
            results.append((_result_prefix + output[12:] + _result_suffix).replace('  ',' '))

    return results





def generate_summaries_or_translations(
    args,
    examples: list,
    out_file: str,
    model_name: str,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    fp16=False,
    task="summarization",
    constraints_list=None,
    lemmatized_constraints_list = None,
    **gen_kwargs,
) -> None:

    fout = Path(out_file).open("w", encoding="utf-8")

    print(f'Decode with {str(model_name)}')


    if args.pl_ckpt == True:


        # model_3 = LRGenerative(hf_name = args.pl_model)
        # # model_3 = model_3.load_from_checkpoint(model_name).reasoner
        # model_3 = model_3.reasoner
        # print(model_3.lm_head.weight)
        # print(f'We use finetuned ckpt of {args.pl_model}')

        # add to device

        ckpt = torch.load(model_name)
        ckpt = ckpt.get('state_dict',ckpt)
        ckpt = OrderedDict([(k.replace('module.reasoner.',''),v) if 'module.' in k else (k.replace('reasoner.',''),v) for k,v in ckpt.items()])
        model_config = AutoConfig.from_pretrained(args.pl_model)

        model = AutoModelForSeq2SeqLM.from_config(model_config)
        model.load_state_dict(ckpt)
        model.to(device)

    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)


    if fp16:
        model = model.half()

    if args.pl_ckpt == True:
        model_name = args.pl_model

    tokenizer = AutoTokenizer.from_pretrained(model_name)


    period_id = [tokenizer.convert_tokens_to_ids('.')]
    if "bart" in args.model_name:
        period_id.append(tokenizer.convert_tokens_to_ids('Ġ.'))
    eos_ids = [tokenizer.eos_token_id] + period_id

    constraints_list = utils_seq2seq.tokenize_constraints(tokenizer, constraints_list)

    # update config with summarization specific params
    use_task_specific_params(model, task)




    demonstration = \
    f"Input: PersonX sneaks into PersonY's room so PersonX feels nervous.\n"\
    "Constraint: and, closet\n"\
    "Output: PersonX sneaks into PersonY's room and sees a closet space, so PersonX feels nervous.\n"\
    f"Input: PersonX sneaks into PersonY's room so PersonX feels nervous.\n"\
    "Constraint: and, furniture, no\n"\
    "Output: PersonX sneaks into PersonY's room and does not find furniture, so PersonX feels nervous.\n"\
    f"Input: PersonX asks what to do so PersonX feels uncertain.\n"\
    "Constraint: and, seek\n"\
    "Output: PersonX asks what to do and seeks suggestions, so PersonX feels uncertain.\n"\
    f"Input: PersonX asks what to do so PersonX feels uncertain.\n"\
    "Constraint: and, help, no\n"\
    "Output: PersonX asks what to do and no one help, so PersonX feels uncertain."





    for batch, cons, lemmatized_cons in tqdm(zip(list(chunks(examples, batch_size)), list(chunks(constraints_list, batch_size)),list(chunks(lemmatized_constraints_list, batch_size)))):
        constraints = init_batch(raw_constraints=cons,
                                 beam_size=args.beam_size,
                                 eos_id=eos_ids)

        batch = [f'Input: {batch[index]} ; Constraint: {lemmatized_cons[index]} ; Output: ' for index in range(len(batch))]
        /test
        if "t5" in model_name:
            # batch = ['generate a sentence with: ' + text + ' </s>' for text in batch]
            batch = [text + ' </s>' for text in batch]

        batch_ids = tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(device)
        input_ids, attention_mask = trim_batch(**batch_ids, pad_token_id=tokenizer.pad_token_id)
        summaries = generate(self=model,
                             input_ids=input_ids,
                             attention_mask=attention_mask,
                             decoder_start_token_id=tokenizer.bos_token_id,
                             min_length=args.min_tgt_length,
                             max_length=args.max_tgt_length,
                             num_beams=args.beam_size,
                             no_repeat_ngram_size=args.ngram_size,
                             length_penalty=args.length_penalty,
                             constraints=constraints,
                             prune_factor=args.prune_factor,
                             sat_tolerance=args.sat_tolerance,
                             beta=args.beta,
                             early_stop=args.early_stop)



        if "t5" in model_name and '<extra_id_0>' in batch[0]:
            dec = tokenizer.batch_decode(summaries, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            dec = filter(batch,dec)
        elif "t5" in model_name and '<extra_id_0>' not in batch[0]:
            dec = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        elif "bart" in model_name:
            dec = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)


        for hypothesis in dec:
            fout.write(hypothesis.strip() + "\n")
            fout.flush()



def run_generate():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str)

    parser.add_argument("--pl_ckpt",    action='store_true')
    parser.add_argument("--pl_model",     type=str,   help = 'original model for reload model')


    parser.add_argument("--input_path", type=str, help="like cnn_dm/test.source")
    parser.add_argument("--save_path", type=str, help="where to save summaries")
    parser.add_argument("--constraint_file", type=str, help="constraint file")
    parser.add_argument("--constraint_file_lemma", type=str, help="lemmatized constraint for ICL")
    parser.add_argument("--reference_path", type=str, required=False, help="like cnn_dm/test_reference_summaries.txt")
    parser.add_argument("--score_path", type=str, required=False, help="where to save the rouge score in json format")
    parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
    parser.add_argument("--task", type=str, default="summarization", help="typically translation or summarization")
    parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
    parser.add_argument('--beam_size', type=int, default=10, help="Beam size for searching")
    parser.add_argument('--ngram_size', type=int, default=3, help='all ngrams can only occur once')
    parser.add_argument('--length_penalty', type=float, default=0.1, help="length penalty for beam search")
    parser.add_argument('--min_tgt_length', type=int, default=0, help="minimum length of target sequence")
    parser.add_argument('--max_tgt_length', type=int, default=128, help="maximum length of target sequence")
    parser.add_argument(
        "--decoder_start_token_id",
        type=int,
        default=None,
        required=False,
        help="decoder_start_token_id (otherwise will look at config)",
    )
    parser.add_argument(
        "--n_obs", type=int, default=-1, required=False, help="How many observations. Defaults to all."
    )
    parser.add_argument('--prune_factor', type=int, default=50, help="fraction of candidates to keep based on score")
    parser.add_argument('--sat_tolerance', type=int, default=2, help="minimum satisfied clause of valid candidates")
    parser.add_argument('--beta', type=float, default=0., help="reward factor for in progress constraint")
    parser.add_argument('--early_stop', type=float, default=None, help="optional early stop if all constraints are satisfied")

    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()


    examples = [x.rstrip() if "t5" in args.model_name else x.rstrip() for x in open(args.input_path).readlines()]



    def lemma(x):
        if "bart" in args.model_name:
            return f' {x}'
        return x

    with open(args.constraint_file, 'r') as f:
        constraints_list = []
        for line in f:
            constraints = []
            for concept in json.loads(line):
                constraints.append([lemma(c) for c in concept])
            constraints_list.append(constraints)

    def strip_to_format(x):
        return x.rstrip().replace('[','').replace(']','').replace('\"','').replace(', not','')

    lemmatized_constraints_list = [strip_to_format(x) for x in open(args.constraint_file_lemma).readlines()]

    if args.n_obs > 0:
        examples = examples[: args.n_obs]



    generate_summaries_or_translations(
        args,
        examples,
        args.save_path,
        args.model_name,
        batch_size=args.bs,
        device=args.device,
        fp16=args.fp16,
        task=args.task,
        constraints_list=constraints_list,
        lemmatized_constraints_list = lemmatized_constraints_list,
        decoder_start_token_id=args.decoder_start_token_id,
    )
    if args.reference_path is None:
        return
    # Compute scores
    score_fn = calculate_bleu_score if "translation" in args.task else calculate_rouge
    output_lns = [x.rstrip() for x in open(args.save_path).readlines()]
    reference_lns = [x.rstrip() for x in open(args.reference_path).readlines()][: len(output_lns)]
    scores: dict = score_fn(output_lns, reference_lns)
    if args.score_path is not None:
        json.dump(scores, open(args.score_path, "w+"))
    return scores


if __name__ == "__main__":
    run_generate()
