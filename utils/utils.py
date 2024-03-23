import argparse
import numpy as np
import random
import os
import torch
from itertools import chain
from torch import nn
import evaluate

def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_file', type=str, default='output.json')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--split', type=str, default='local', choices=['train', 'validation', 'test', 'local'])
    parser.add_argument('--in_file', type=str, default='data/xsum_test_keyword.json')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--resume_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=None)
    parser.add_argument('--use_cpmi', action='store_true')
    parser.add_argument('--domain_type', choices=['none',
                                                  'prompt_first', 'prompt_keyword', 'prompt_random', 'prompt_random_sentence', 'prompt_key_sentence', 
                                                  'prompt_keyword_reverse', 'prompt_first_reverse', 'prompt_random_reverse', 'prompt_random_sentence_reverse', 'prompt_key_sentence_reverse' 
                                                  ], default='none')
    parser.add_argument('--only_decoder', action='store_true')
    parser.add_argument('--use_language_model', action='store_true')
    parser.add_argument('--lmda', type=float, default=6.5602e-2)
    parser.add_argument('--tau', type=float, default=3.5987)
    parser.add_argument('--eps', type=float, default=1e-10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--run_type', choices=['beam', 'cpmi', 'ours'], default='beam')
    parser.add_argument('--model', choices=['bart', 'pegasus'], default='bart')
    parser.add_argument('--prompt', type=str, default='in summary')

    return parser


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory) 
    except OSError:
        print('Error: Creating directory. '+ directory)


def set_seed(seed):

    print('setting seed:', seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


"""
https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py#L516
"""
def group_texts(examples):
    max_length = 2048 # 1024
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // max_length) * max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = logits.permute(0, 2, 1)
    loss = nn.CrossEntropyLoss()
    result = loss(logits[:, : :, :-1], labels[:, 1:])
    result = e**result
    return result


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    result = {'ppl': sum(preds)/len(preds)}
    return result
