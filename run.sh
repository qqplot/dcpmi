#!/bin/bash

# Beam 
python run_batch.py --output_file bart_beam.json --gpu_id 0 --run_type beam --batch_size 2

# CPMI
python run_batch.py --output_file bart_cpmi.json --gpu_id 2 --run_type cpmi --batch_size 2


# Ours
model=bart                   # bart pegasus
domain_type=prompt_keyword 
prompt="in summary"
lmda=0.065602   # 0.065602 0.074534 
tau=3.5987      # 3.5987 3.304358 
run_type=ours   # cpmi ours
batch_size=2
	
srun python run_batch.py --output_file "${model}_${run_type}_${domain_type}.json" --gpu_id 0 --use_cpmi \
                         --use_language_model --domain_type ${domain_type} --prompt "${prompt}" --run_type ${run_type} --model ${model} --batch_size $batch_size \
                         --lmda ${lmda} --tau ${tau} \
                         --in_file data/xsum_test_keyword.json

