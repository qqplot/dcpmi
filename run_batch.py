import os
import json, random
import torch
import numpy as np

from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline, AutoConfig, AutoModelForCausalLM, PegasusForConditionalGeneration, PegasusTokenizer
from datasets import load_dataset
from evaluate import load
from utils import init_parser, set_seed, generic_text_predictions, createFolder
# from alignscore import AlignScore
import sys, logging
from torch.utils.data import DataLoader
from datetime import datetime

def parse_output_filename(args):
    output_file = args.output_file
    output_file = "".join(output_file.split('.')[:-1])
    return output_file


def batch_writer(current_datetime_str, output_file, doc_id, summary, gold, source):

    for i in range(len(doc_id)):
        output_dict_example = {
            "id" : doc_id[i],
            "predicted" : summary[i],
            "gold" : gold[i],
            "source" : source[i],
        }
        with open(f"results/{current_datetime_str}_{output_file}.json", "a") as _jsonl_file:
            _jsonl_file.write(json.dumps(output_dict_example))
            _jsonl_file.write("\n")
    return


def make_domain_prompt(args, keyword, input_text):
    if args.domain_type == 'keyword':
        topic = [f"{t}" for t in keyword]
    elif args.domain_type == 'prompt_first':
        topic = [f"{args.prompt} " + t.split("\n")[0] for t in input_text]
    elif args.domain_type == 'prompt_keyword':
        topic = [f"{args.prompt} {t}" for t in keyword]
    elif args.domain_type == 'prompt_keyword_reverse':
        topic = [f"{t} {args.prompt}" for t in keyword]            
    elif 'prompt_random' in args.domain_type:
        topic = []
        for text in input_text:
            sentences = text.split('\n')
            sentences = [s.split(' ') for s in sentences]
            words = []
            for s in sentences:
                words.extend(s)
            selected_words = np.random.choice(words, size=3, replace=False)
            if 'reverse' in args.domain_type:
                topic.append(f"""{' '.join(selected_words)} {args.prompt}""")
            else:
                topic.append(f"""{args.prompt} {' '.join(selected_words)}""")                    
    elif 'prompt_random_sentence' in args.domain_type:
        topic = []
        for text in input_text:
            sentences = text.split('\n')
            sentences = np.random.choice(sentences, size=1, replace=False)
            if 'reverse' in args.domain_type:                    
                topic.append(f"{sentences[0]} {args.prompt}")
            else:
                topic.append(f"{args.prompt} {sentences[0]}")
    elif 'prompt_key_sentence' in args.domain_type:
        topic = []
        for i, text in enumerate(input_text):
            sentences = text.split('\n')
            keywords = keyword[i].split(', ')
            if len(keywords) == 0:
                assert True, f"please check keyword - id :{doc_id[i]}"
            keyword = keywords.pop(0)
            key_sentence = ""
            for seq in sentences:
                if keyword in seq.lower():
                    key_sentence = seq
                    break           
            if key_sentence == "" and len(keywords) > 0:
                for seq in sentences:
                    if keyword in seq.lower():
                        key_sentence = seq
                        print(f"{keyword}, {seq}")
                        break   
            if key_sentence == "":
                key_sentence = sentences[0]
            if 'reverse' in args.domain_type:                      
                topic.append(f"{key_sentence} {args.prompt}")
            else:
                topic.append(f"{args.prompt} {key_sentence}")
    else:
        topic = ''

    return topic    


def main(args):

    output_file = parse_output_filename(args)
    current_datetime = datetime.now()
    current_datetime_str = current_datetime.strftime("%Y-%m-%dT%H:%M:%S")

    ### SETTING LOGGER
    createFolder('logs')
    createFolder('results')    
    logging.basicConfig(
        filename=f"logs/{current_datetime_str}_{output_file}.log", filemode="w",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(vars(args))    
    logger.info(f"run_type: {args.run_type}, use_cpmi: {args.use_cpmi}, use_language_model: {args.use_language_model}")

    device = torch.device(f'cuda:{args.gpu_id}')
    set_seed(args.seed)

    if args.split == "local":        
        if args.end_index is None:
            dataset = load_dataset('json', data_files=args.in_file, split=f"train[{args.resume_index}:]")
        else:
            dataset = load_dataset('json', data_files=args.in_file, split=f"train[{args.resume_index}:{args.end_index}]")
    else:
        if args.end_index is None:
            dataset = load_dataset("EdinburghNLP/xsum", split=f"{args.split}[{args.resume_index}:]")
        else:
            dataset = load_dataset("EdinburghNLP/xsum", split=f"{args.split}[{args.resume_index}:{args.end_index}]")
    
    logger.info(f"len(dataset): {len(dataset)}")

    if args.model == 'bart':
        logger.info("loading bart...")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-xsum", forced_bos_token_id=0)
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-xsum", output_attentions=True)
        model.to(device)
        logger.info("bart loaded.")
    elif args.model == 'pegasus':
        logger.info("loading pegasus...")
        tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum", forced_bos_token_id=0)
        model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum", output_attentions=True)
        model.to(device)
        logger.info("pegasus loaded.")
    else:
        assert True, "Please check your 'model' option..."
    
    user_kwargs = None
    if args.use_cpmi:
        user_kwargs = {}

        language_model = None
        if args.use_language_model:
            if args.model == 'bart':
                language_model_config = AutoConfig.from_pretrained('qqplot23/xsum-gpt2-long')
            elif args.model == 'pegasus':
                language_model_config = AutoConfig.from_pretrained('qqplot23/xsum-gpt2-long-pegasus')
            
            language_model = AutoModelForCausalLM.from_config(config=language_model_config)

            embedding_size = model.get_input_embeddings().weight.shape[0]
            lang_emb_size = language_model.get_input_embeddings().weight.shape[0]
            if lang_emb_size != embedding_size:
                language_model.resize_token_embeddings(embedding_size)

            language_model.to(device)
            logger.info("language_model loaded.")

        user_kwargs["language_model"] = language_model                
        user_kwargs["lmda"] = args.lmda
        user_kwargs["tau"] = args.tau
        user_kwargs["eps"] = args.eps
        user_kwargs["only_decoder"] = args.only_decoder
        user_kwargs["run_type"] = args.run_type
                        
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    for i, input_item in enumerate(tqdm(test_dataloader, desc="Predicting...")):
        input_text = input_item["document"] # batch 
        doc_id = input_item["id"]           # batch 
        gold = input_item["summary"]        # batch 

        topic = make_domain_prompt(args, keyword=input_item['keyword'], input_text=input_text)

        if user_kwargs is not None:
            user_kwargs["prefix"] = topic

        summary = generic_text_predictions(
            args,
            model,
            tokenizer,
            input_text,
            device,
            user_kwargs=user_kwargs,
        )

        output_dict_example = {
            "output_file" : output_file,
            "doc_id" : doc_id,
            "summary" : summary,            
            "gold" : gold,
            "source" : input_text,
        }
        batch_writer(current_datetime_str=current_datetime_str, **output_dict_example)
    return


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    
    main(args)