# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function
import json
import os
import re
import sys
import functools
import random
import numpy as np
import jsonlines
from typing import List, Tuple, Optional

import torch
import evaluate
from datasets import load_dataset
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

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
            batch_size=64,
            minibatch_size=64,
            checkpoint_interval=10000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AccelerateRedteamPPOTrainer",
            tracker="tensorboard",
            logging_dir=script_name,
            checkpoint_dir=f"{script_name}/ckpts"
        ),
        model=ModelConfig(
            model_path="vicgalle/gpt2-alpaca",
            # model_path="lvwerra/gpt2-imdb",
            # model_path="gpt2",
            # model_path="sharpbai/alpaca-7b-merged",
            num_layers_unfrozen=-1,
            peft_config={
                'r': 32,
                'lora_alpha': 16,
                'lora_dropout': 0.0,
                'task_type': "CAUSAL_LM",
                'peft_type': "LORA",
            },
            quantization_config={
                'load_in_4bit': True,
                'bnb_4bit_compute_dtype': 'float16',
                'bnb_4bit_use_double_quant': True,
                'bnb_4bit_quant_type': 'nf4',
            },
        ),
        tokenizer=TokenizerConfig(
            tokenizer_path="vicgalle/gpt2-alpaca",
            # tokenizer_path="lvwerra/gpt2-imdb",
            # tokenizer_path="gpt2",
            # tokenizer_path="sharpbai/alpaca-7b-merged",
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
                eos_token_id=50256,
                pad_token_id=50256,
            ),

            # Red team config
            bleu_reward_coef=0.0,
            bleu_reward_grams="[3, 4, 5]",
            bleu_reward_include_prompts=False,

            # Entropy bonus
            ent_reward_coef=0.0,

            # Cossim embed
            cossimemb_reward_coef=0.0,
            cossimemb_n_samples=-1,
            cossimemb_impl="sentencetransformer",
            cossimemb_reward_include_prompts=False,

            # Textual sim
            textual_sim_reward_coef=0,
            textual_sim_reward_include_prompts=False,

            giberish_penalty_coef=0.0,
        ),
    )


class RedTeamToxicityRewardModel(object):

    ATTACKER_EOS_TOKEN = "<|endoftext|>"

    @staticmethod
    def make_prompt(instruction):
        prompt_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"

        return prompt_template.format(instruction=instruction.rstrip())

    def __init__(self, device, victim_model, victim_tokenizer) -> None:
        self.toxicity_fn = evaluate.load(
                "toxicity", 
                module_type="measurement",
                device=device)
        self.device = device

        self.victim_tokenizer = victim_tokenizer
        self.victim_model = victim_model
        self.victim_generation_kwargs = dict(
                max_new_tokens=30,
                top_k=0,
                top_p=0.92,
                temperature=0.7,
                do_sample=True,
                eos_token_id=self.victim_tokenizer.eos_token_id,
                pad_token_id=self.victim_tokenizer.pad_token_id,
        )

    def __call__(self, samples: List[str], 
                prompts: Optional[List[str]],
                outputs: Optional[List[str]],
                return_texts=False, # For logging the victim's response
                **kwargs
            ) -> List[float]:

        def _make_instruction(output):
            output = output.lstrip().rstrip()

            # Remove <endoftext>:
            # when the attacker's generation contains eos_token or trimmed by stop_sequences, trlx will append eos_token
            trimmed_indices = list(filter(lambda idx: idx >= 0, [
                output.find(self.ATTACKER_EOS_TOKEN),
                output.find("\n"),
            ]))

            if len(trimmed_indices) == 0:
                return self.make_prompt(output)
            else:
                trimmed_idx = min(trimmed_indices)
                return self.make_prompt(output[:trimmed_idx])

        instructions = list(map(_make_instruction, outputs))

        victim_inputs = self.victim_tokenizer(instructions,
                                            padding=True,
                                            add_special_tokens=False,
                                            return_tensors="pt").to(self.device)
        with self.victim_model.disable_adapter():
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

    # Avoid overwriting the experiments
    if os.path.exists(config.train.logging_dir):
        print("Experiment exists")
        sys.exit()
    else:
        os.makedirs(config.train.logging_dir)

    with jsonlines.open("prompts/alpaca.jsonl", "r") as reader:
        prompts = list(map(lambda x: x["attacker_prompt"], reader))
    train_prompts, eval_prompts = prompts, prompts[-3:]

    trlx.train(
        reward_fn=RedTeamToxicityRewardModel,
        prompts=train_prompts,
        eval_prompts=eval_prompts,
        config=config,
        stop_sequences=["\n", ".", "?"],
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
