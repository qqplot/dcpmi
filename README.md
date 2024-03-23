# Mitigating Hallucination in Abstractive Summarization with Domain-Conditional Mutual Information

This is the official repo of the paper "[Mitigating Hallucination in Abstractive Summarization with Domain-Conditional Mutual Information](https://openreview.net/forum?id=N5gW9kxJ7Z)" We provide datasets for reproducing our results on XSUM. We do not guarantee exact reproducibility, as library versions and GPUs may cause small differences, but these should be extremely minor.


## Abstract
A primary challenge in abstractive summarization is hallucination---the phenomenon where a model generates plausible text that is absent in the source text. We hypothesize that the domain (or topic) of the source text triggers the model to generate text that is highly probable in the domain, neglecting the details of the source text. To alleviate this model bias, we introduce a decoding strategy based on domain-conditional pointwise mutual information. This strategy adjusts the generation probability of each token by comparing it with the token's marginal probability within the domain of the source text. According to evaluation on the XSUM dataset, our method demonstrates improvement in terms of faithfulness and source relevance.


## Installation 
Our code is based on Huggingface's `transformers>=4.35.0`.
The following files are primarily modified.

* [utils.py](https://github.com/qqplot/dcpmi/blob/main/transformers/src/transformers/generation/utils.py)
* [modeling_bart.py](https://github.com/qqplot/dcpmi/blob/main/transformers/src/transformers/models/bart/modeling_bart.py)
* [modeling_pegasus.py](https://github.com/qqplot/dcpmi/blob/main/transformers/src/transformers/models/pegasus/modeling_pegasus.py)


```bash
conda create -n dcpmi python=3.8 -y

conda activate dcpmi
pip install torch torchvision torchaudio

cd transformers
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.

pip install datasets evaluate rouge-score nltk
```



## Run
Please refer to the code example below for instructions on how to run the code.

```sh
# Beam 
python run_batch.py --output_file bart_beam.json --gpu_id 0 --run_type beam --batch_size 2

# CPMI
python run_batch.py --output_file bart_cpmi.json --gpu_id 2 --run_type cpmi --batch_size 2

# Ours
python run_batch.py --output_file bart_ours.json --gpu_id 2 --use_cpmi --run_type ours --use_language_model --domain_type "prompt_keyword" --prompt "in summary" --batch_size 2
```


## Evaluation

To perform evaluation, you need to install the metrics.
- [AlignScore](https://github.com/yuh-zha/AlignScore)
- [BARTScore](https://github.com/neulab/BARTScore)


```bash
python evaluation.py --input_file "./results/2024-03-24T02:54:11_bart_ours.json" --output_file "eval.json" --batch_size 64 
```


# Cite Our Work
```
@inproceedings{
chae2024mitigating,
title={Mitigating Hallucination in Abstractive Summarization with Domain-Conditional Mutual Information},
author={Kyubyung Chae and Jaepill choi and Yohan Jo and Taesup Kim},
booktitle={2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics},
year={2024},
url={https://openreview.net/forum?id=N5gW9kxJ7Z}
}
```