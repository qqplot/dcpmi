# Mitigating Hallucination in LLM with Uncertainty-aware scoring


## Installation 
아직 환경 설정을 테스트해보지는 못해서 안 돌아갈 수도 있습니다.
(`requirements.txt`에 있는 패키지를 모두 설치할 필요 없습니다.)
아래 코드로 설치해보고, 안 되면 연락바랍니다.

```bash
conda create -n dcpmi python=3.8 -y

conda activate dcpmi
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip install datasets evaluate rouge-score nltk
```

### Install Alignscore
[여기](https://github.com/yuh-zha/AlignScore/tree/main)에서 repo를 받은 뒤 아래 코드를 실행한다.

```
cd AlignScore
pip install .
python -m spacy download en_core_web_sm
```


# Run
Huggingface의 `transformers>=4.35.0`를 기반으로 작성되었습니다.


아래 파일이 주로 수정되었습니다.
* [utils.py](https://github.com/qqplot/dcpmi/blob/main/transformers/src/transformers/generation/utils.py)
* [modeling_bart.py](https://github.com/qqplot/dcpmi/blob/main/transformers/src/transformers/models/bart/modeling_bart.py)
* [modeling_pegasus.py](https://github.com/qqplot/dcpmi/blob/main/transformers/src/transformers/models/pegasus/modeling_pegasus.py)


`utils.py`에 [3133th line](https://github.com/qqplot/dcpmi/blob/main/transformers/src/transformers/generation/utils.py#L3133)을 살펴보세요.
수정된 부분은 qqplot을 검색해서 찾으시면 됩니다.

실행방법은 아래 코드를 참고해주세요.

```sh
# Beam 
python run.py --output_file output_beam.json --gpu_id 0 --run_type beam

# CPMI
python run.py --output_file output_cpmi.json --gpu_id 0 --use_cpmi --run_type cpmi 

# Ours
python run.py --output_file output_ours.json --gpu_id 0 --use_cpmi --run_type ours --alpha 0.6 --beta 0.01 --use_language_model --soft_uncertainty_weight

# Search hyperparameters
python search_hyperparameters.py --output_file ours.csv --gpu_id 0 --use_cpmi --run_type ours --use_language_model --soft_uncertainty_weight
```

<!-- # git clone https://github.com/huggingface/transformers.git
# cd transformers
# pip install -e .
# pip install -r requirements.txt
# python -m spacy download en_core_web_sm -->

<!-- It will install a customized version of [HuggingFace transformers](https://github.com/dakinggg/transformers/tree/my_branch) with some edits to the beam search code.  -->

