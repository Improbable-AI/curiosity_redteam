Curiosity-driven Red-teaming for Large Language Models
=

This repository contains the official implementation for [ICLR'24 paper](https://openreview.net/pdf?id=4KqkizXgXU), Curiosity-driven Red-teaming for Large Language Models.

# Abstract
Large language models (LLMs) hold great potential for various natural language applications but risk generating incorrect or toxic content. To probe when an LLM generates unwanted content, the current paradigm is to recruit human testers to create input prompts (i.e., test cases) designed to elicit unfavorable responses from LLMs. This procedure is called red teaming. However, relying solely on human testers can be both expensive and time-consuming. Recent works automate red teaming by training LLMs (i.e., red team LLMs) with reinforcement learning (RL) to maximize the chance of eliciting undesirable responses (i.e., successful test cases) from the target LLMs being evaluated. However, while effective at provoking undesired responses, current RL methods lack test case diversity as RL-based methods tend to generate a small number of successful test cases once found (i.e., high-precision but low diversity). To overcome this limitation, we propose using curiosity-driven exploration optimizing for novelty to train red team models for generating a set of diverse and effective test cases. We evaluate our method by performing red teaming against LLMs in text continuation and instruction following tasks. Our experiments show that curiosity-driven exploration achieves greater diversity in all the experiments compared to existing RL-based red team methods while maintaining effectiveness. Remarkably, curiosity-driven exploration also enhances the effectiveness when performing red teaming in instruction following test cases, generating more successful test cases. Finally, we demonstrate that the proposed approach successfully provokes toxic responses from LLaMA2 model that has undergone substantial finetuning based on human preferences.

# Installation
```
conda create -n redteam python=3.10
git clone git@github.com:Improbable-AI/curiosity_redteam.git
cd custom_trlx
pip install -e .
cd ..
pip install -r requirements.txt
export PYTHONPATH=$(pwd):$PYTHONPATH
```

# Experiments

The following are the basic steps to launch the red-teaming experiments. The results will be put at `results` directory. Note that you need to install `tensorboard` to log the results correctly. If the result directory exists, the experiment will be terminated. This avoids overwriting the existing results, but could be annoying when you are experimenting a new feature. You may turn on debug mode by adding to any of the following commands `--debug`. This will put the results in `debug` with randomly generated numbers so that you don't have to delete the result directory every time.

You can play the hyperparameters in each script in the `experiments` directory.

## Text continuation

In this experiment, we train a GPT2 model to elicit toxic continuation from another GPT2 model. 

**Hardware requirement:** 24GB VRAM

**Script:** `python experiments/imdb_toxicity_response/run_ppo.py --mode local --gpus 0`


## Instruction-following

**Hardware requirement:** 48GB VRAM

**Script:** 
- `python experiments/alpaca_toxicity/run_ppo_gpt2_gpt2_peft.py --mode local --gpus 0`
- `python experiments/databrick_toxicity/run_ppo_gpt2_dolly_peft.py --mode local --gpus 0`
- `python experiments/databrick_toxicity/run_ppo_gpt2_llama_safetyguard_peft.py --mode local --gpus 0`
- `python experiments/databrick_toxicity/run_ppo_gpt2_vicuna_peft.py --mode local --gpus 0`


## Text-to-image 

**Hardware requirement:** 48GB VRAM

**Script:** `python experimentsi/sd_nsfw/run_ppo_gpt2_sd.py --mode local --gpus 0`


## Citation
```latex
@inproceedings{
hong2024curiositydriven,
title={Curiosity-driven Red-teaming for Large Language Models},
author={Hong, Zhang-Wei and Shenfeld, Idan and Wang, Tsun-Hsuan and Chuang, Yung-Sung and Pareja, Aldo and Glass, James and Srivastava, Akash and Agrawal, Pulkit},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=4KqkizXgXU}
}
```
