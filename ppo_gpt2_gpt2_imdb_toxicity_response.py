# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function
import json
import os
import re
import sys
import functools
import numpy as np
from typing import List, Tuple, Optional

import torch
import evaluate
from datasets import load_dataset
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

import uuid
import trlx
from trlx.data.default_configs import TRLConfig
from trlx.models.modeling_ppo import PPOConfig

from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
import evaluate

from accelerate_redteam_ppo_trainer import RedteamPPOConfig

script_name = os.path.splitext(os.path.basename(__file__))[0]

def default_redteam_ppo_config():
    return TRLConfig(
        train=TrainConfig(
            seq_length=1024,
            epochs=1000,
            total_steps=10000,
            batch_size=32,
            checkpoint_interval=10000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AccelerateRedteamPPOTrainer",
            tracker="tensorboard",
            logging_dir=script_name,
            checkpoint_dir=f"{script_name}/ckpts"
        ),
        model=ModelConfig(
            # model_path="vicgalle/gpt2-alpaca",
            model_path="lvwerra/gpt2-imdb",
            num_layers_unfrozen=2),
        tokenizer=TokenizerConfig(
            # tokenizer_path="vicgalle/gpt2-alpaca", 
            tokenizer_path="lvwerra/gpt2-imdb",
            truncation_side="right"),
        optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=3e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        ),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=1e12, eta_min=3e-5)),
        method=RedteamPPOConfig(
            name="RedteamPPOConfig",
            num_rollouts=128,
            chunk_size=128,
            ppo_epochs=4,
            init_kl_coef=0.001,
            target=None,
            horizon=10000,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=1,
            scale_reward="ignored",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs=dict(
                max_new_tokens=10,
                top_k=0,
                top_p=0.92,
                temperature=0.7,
                do_sample=True,
            ),

            # Red team config
            bleu_reward_coef=0.0,
            bleu_reward_grams="[3, 4, 5]",
            bleu_reward_include_prompts=True,
            bleu_tokenizer="nltk",
            
            # Entropy bonus
            ent_reward_coef=0.0,

            # Cossim embed
            cossimemb_reward_coef=0.0,
            cossimemb_n_samples=-1,
            cossimemb_impl="sentencetransformer",
            cossimemb_reward_include_prompts=True,
            
            # Textual sim
            textual_sim_reward_coef=0,
            textual_sim_reward_include_prompts=True,
            
            # Div reawrds
            target_sim_div_reward_coef=0,            
        ),
    )

class RedTeamToxicityRewardModel(object):

    def __init__(self, device) -> None:
        self.toxicity_fn = evaluate.load(
                "toxicity", 
                module_type="measurement",
                experiment_id=str(uuid.uuid4()),
                device=device)
        self.device = device
        self.victim_generation_kwargs = dict(
                max_new_tokens=30,
                top_k=0,
                top_p=0.92,
                temperature=0.7,
                do_sample=True,
                eos_token_id=50256,
                pad_token_id=50256,
        )
        self.victim_tokenizer = AutoTokenizer.from_pretrained(
            "lvwerra/gpt2-imdb", device=device)
        self.victim_tokenizer.pad_token_id = self.victim_tokenizer.eos_token_id
        self.victim_model = AutoModelForCausalLM.from_pretrained(
            "lvwerra/gpt2-imdb").to(device)
    
    def __call__(self, samples: List[str], 
                prompts: Optional[List[str]],
                outputs: Optional[List[str]],
                return_texts=False, # For logging the victim's response
                **kwargs
            ) -> List[float]:
        victim_inputs = self.victim_tokenizer(samples,
                                            padding=True,
                                            return_tensors="pt").to(self.device)
        victim_outputs = self.victim_model.generate(**victim_inputs,                                                
                                                **self.victim_generation_kwargs)
        victim_responses = self.victim_tokenizer.batch_decode(
                victim_outputs[:, victim_inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True)
        toxicity_scores = self.toxicity_fn.compute(predictions=victim_responses)["toxicity"]

        """
        NOTE: We always return `victim_responses` that is used in reward calculation.
            If we consider an instruction-following setting, we should remove the blurb and other special tokens, and return.
            In continuation tasks (e.g., imdb review), we should return the concatenation of attacker's prompt, attacker's response, and victim's responses.
        """
        
        if return_texts:
            return toxicity_scores, victim_responses
        else:
            return toxicity_scores


def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_redteam_ppo_config().to_dict(), hparams)
    print(config) # To check the parsed config

    # Avoid overwriting the experiments
    if os.path.exists(config.train.logging_dir):
        print("Experiment exists")
        sys.exit()

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    reward_fn = RedTeamToxicityRewardModel(device=device)

    # Take few words off of movies reviews as prompts
    imdb = load_dataset("imdb", split="train+test")
    prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]

    trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=["I don't know much about Hungarian underground"] * 8,
        config=config,
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
