import random
import re
from typing import List

import numpy as np
import torch

def preprocess_function(examples, prefix=''):
    if prefix == '':
        return examples    
    inputs = [doc + ' ' + p for doc, p in zip(examples, prefix)]
    return inputs


def calculate_entropy(log_probabilities):
    """
    Calculate entropy using natural log probabilities.

    Parameters:
    - log_probabilities: A list of natural log probabilities for different outcomes.

    Returns:
    - entropy: The calculated entropy.
    """
    log_probabilities = log_probabilities.detach().cpu()
    log_probabilities = torch.where(torch.isnan(log_probabilities) | torch.isinf(log_probabilities), torch.tensor(0.0), log_probabilities)
    probabilities = torch.exp(log_probabilities)
    entropy = -torch.sum(probabilities * log_probabilities)
    return entropy.item()  # Convert the result to a Python float


def generic_text_predictions(
    args,
    model,
    tokenizer,
    batch,
    device,
    user_kwargs=None,
):
    
    if user_kwargs is not None:
        decoder_prefix = None
        if user_kwargs["prefix"] != '':
            user_kwargs["prefix"] = [t.replace(',', '') for t in user_kwargs["prefix"]]
            dct_prefix = tokenizer(user_kwargs['prefix'], return_tensors="pt", verbose=False, padding=True, add_special_tokens=True)            
            decoder_prefix = dct_prefix["input_ids"].to(device)
        user_kwargs["decoder_prefix"] = decoder_prefix # batch

        if not user_kwargs["only_decoder"]:
            batch = preprocess_function(batch, prefix=user_kwargs["prefix"])

    if args.model == 'bart':
        model_max_length = 1024
        num_beams = 6
        max_length = 62
    elif args.model == 'pegasus':
        model_max_length = 512
        num_beams = 8
        max_length = 64

    dct = tokenizer(batch, max_length=model_max_length, return_tensors="pt", truncation=True, padding=True)

    input_ids = dct["input_ids"].to(device)
    attention_mask = dct["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        min_length=5,            
        max_length=max_length,   
        early_stopping=True,     
        no_repeat_ngram_size=3, 
        length_penalty=2.0, 
        temperature=1.0, 
        top_k=50,  
        top_p=1.0, 
        num_beams=num_beams,  
        renormalize_logits=True, 
        do_sample=False,  
        output_scores=True,
        return_dict_in_generate=True,
        user_kwargs=user_kwargs,   # user defined
    )

    sequences = outputs[0]    
    outtext = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    return outtext


def erase_prefix(sequences, prefix):
    out = []
    for seq, p in zip(sequences, prefix):
        out.append(seq[len(p):])
    return out

    