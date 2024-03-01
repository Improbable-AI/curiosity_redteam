import contextlib
import inspect
import json
import os
import sys
import functools
from abc import abstractmethod
from contextlib import contextmanager
from time import time
from typing import Dict, List, Optional, Tuple, Any

from dataclasses import dataclass, field
from torchtyping import TensorType

import yaml
import ray
import torch
from accelerate import Accelerator  # type: ignore
from ray.air import session
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer

import torch.nn as nn
import torch.nn.functional as F

import trlx.utils.logging as logging
from trlx.data.configs import TRLConfig
from trlx.pipeline import MiniBatchIterator
from trlx.trainer import BaseRLTrainer, register_trainer
from trlx.utils import (
    filter_non_scalars,
    get_distributed_config,
    get_git_tag,
    get_optimizer_class,
    get_scheduler_class,
    significant,
)
from trlx.utils.modeling import (
    flatten_dict,
    freeze_bottom_causal_layers,
    freeze_bottom_seq2seq_layers,
    gather_dict,
)

import nltk

import json
import os
import uuid
from time import time
from typing import Callable, List

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import trlx.utils.logging as logging
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.configs import TRLConfig
from trlx.data.ppo_types import PPORLBatch, PPORLElement
from trlx.models.modeling_ppo import (
    AdaptiveKLController,
    AutoModelForCausalLMWithHydraValueHead,
    AutoModelForSeq2SeqLMWithHydraValueHead,
    FixedKLController,
)
from trlx.pipeline.offline_pipeline import PromptPipeline
from trlx.pipeline.ppo_pipeline import PPORolloutStorage
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from trlx.utils import Clock, infinite_dataloader
from trlx.utils.modeling import RunningMoments, gather_dict, logprobs_of_labels

from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer
from trlx.models.modeling_ppo import PPOConfig

from trlx.data.method_configs import MethodConfig, register_method

from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

import pandas as pd

from self_bleu import SelfBleuReward
from sentence_embed import CosineSentenceEmbeddingReward

from clean_reward import GiberishPenalty

from PIL.Image import Image

logger = logging.get_logger(__name__)

class TextCSVLogger(object):

    def __init__(self, log_dir, output_filename):
        self.log_dir = log_dir
        self.output_filename = output_filename
        self.base_filename = os.path.splitext(output_filename)[0]
        self.iter_count = 0

    def _make_image_paths(self, n):
        base_dir = os.path.join(self.log_dir, self.base_filename, "images", f"{self.iter_count:05d}")
        os.makedirs(base_dir, exist_ok=True)
        return [f"{base_dir}/{i:03d}.png" for i in range(n)]

    def log(self, attacker_prompts, attacker_outputs, victim_outputs, scores):
        if isinstance(victim_outputs[0], Image):
            image_paths = self._make_image_paths(len(victim_outputs))
            for image_path, image in zip(image_paths, victim_outputs):
                image.save(image_path)
            victim_outputs = image_paths

        str_df = pd.DataFrame({
            "attacker's prompts": attacker_prompts,
            "attacker's responses": attacker_outputs,
            "victim's responses": victim_outputs,
            "score": scores,
            "iter": self.iter_count,
        })        
        str_df.to_csv(
            os.path.join(self.log_dir, self.output_filename),
            mode='w' if (self.iter_count == 0) else 'a', header=(self.iter_count == 0),
            sep="\t")
        self.iter_count += 1

class TextCSVLoggerWithTimestamp(object):

    def __init__(self, log_dir, output_filename):
        self.log_dir = log_dir
        self.output_filename = output_filename
        self.iter_count = 0 

    def log(self, attacker_prompts, attacker_outputs, victim_outputs, scores, timestamp):
        str_df = pd.DataFrame({
            "attacker's prompts": attacker_prompts,
            "attacker's responses": attacker_outputs,
            "victim's responses": victim_outputs,
            "score": scores,
            "iter": self.iter_count,
            "timestamp": timestamp,
        })        
        str_df.to_csv(
            os.path.join(self.log_dir, self.output_filename),
            mode='w' if (self.iter_count == 0) else 'a', header=(self.iter_count == 0),
            sep="\t")
        self.iter_count += 1

@dataclass
@register_method
class RedteamPPOConfig(PPOConfig):
    
    '''
    BLEU rewards configuration
    '''
    bleu_reward_coef: float = -0.5 # NOTE: must be negative since we want to minimize overlap
    bleu_reward_grams: str = "[3, 4, 5]" # NOTE: accelerate tracker cannot log list arguments
    bleu_reward_include_prompts: bool = False # Including prompts in continuation tasks
    bleu_tokenizer: str = "nltk"
    bleu_n_samples: int = -1

    '''
    Entropy bonus configuration (i.e., KL penalty to uniform distribution)
    '''
    ent_reward_coef: float = 0.0

    '''
    Sentence embedding bonus
    '''
    cossimemb_reward_coef: float = 0.0
    cossimemb_n_samples: int = -1
    cossimemb_impl: str = "huggingface"
    cossimemb_reward_include_prompts: bool = True
    cossimemb_model_device: str = "default" 
    
    '''
    Textual similarity reward (between attacker's prompts and attacker's responses)
    '''
    textual_sim_reward_coef: float = 0.0
    textual_sim_reward_include_prompts: bool = False
    
    '''
    Target model's batch embedding diversity
    '''
    target_sim_div_reward_coef: float = 0.0
    
    '''
    GiberishPenalty
    '''
    giberish_penalty_coef: float = 0.0
    giberish_model_device: str = "default" # same as attacker
    
    '''
    Reward model device
    '''
    reward_model_device_offset: int = 0


@register_trainer
class AccelerateRedteamPPOTrainer(AcceleratePPOTrainer):

    def __init__(self, config: TRLConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._setup_redteam(config)

    def _setup_redteam(self, config):
        if inspect.isclass(self.reward_fn) or isinstance(self.reward_fn, functools.partial):
            self.reward_fn = self.reward_fn(self.accelerator.device, self.model.base_model, self.tokenizer)
        
        if self.config.method.bleu_tokenizer == "nltk":
            bleu_tokenizer = nltk.word_tokenize
            print(f"BLEU tokenizer: {bleu_tokenizer}")
        elif self.config.method.bleu_tokenizer == "model":
            print(f"BLEU tokenizer: {self.tokenizer}")
            bleu_tokenizer = lambda x: self.tokenizer.batch_decode(self.tokenizer(x, return_tensors="pt")["input_ids"][0].unsqueeze(1))
        
        self.bleu_reward_module = SelfBleuReward(
            grams=eval(config.method.bleu_reward_grams),
            sample_size=config.method.bleu_n_samples,
            tokenizer=bleu_tokenizer,
        )
        
        self.cossimemb_reward_module = CosineSentenceEmbeddingReward(
            n_samples=config.method.cossimemb_n_samples,
            impl=config.method.cossimemb_impl,
            device=(self.accelerator.device if config.method.cossimemb_model_device == "default" else config.method.cossimemb_model_device)
        )
        
        if self.config.method.giberish_penalty_coef != 0:
            self.giberish_penalty_penalty_module = GiberishPenalty((self.accelerator.device if config.method.giberish_model_device == "default" else config.method.giberish_model_device))
    
        self.train_text_logger = TextCSVLogger(self.accelerator.project_dir, "train.csv")
        self.eval_text_logger = TextCSVLogger(self.accelerator.project_dir, "eval.csv")
        self.history_scores = []

    @torch.inference_mode()
    def _process_element(self, ppo_rl_elements, samples, batch, prompt_tensors, sample_outputs, scores, scores_mask, device):
        # Precompute logprobs, values
        if self.config.model.model_arch_type == "seq2seq":
            attention_mask = batch.attention_mask.to(device)
            prompt_tensors = batch.input_ids.to(device)
            decoder_attention_mask = sample_outputs.not_equal(self.tokenizer.pad_token_id)
            decoder_attention_mask[:, 0] = 1
            batch_size = sample_outputs.shape[0]
            with torch.no_grad():
                attention_mask_arg = attention_mask if batch_size != 1 else None
                outputs = self.model(
                    input_ids=prompt_tensors,
                    attention_mask=attention_mask_arg,
                    decoder_input_ids=sample_outputs,
                    decoder_attention_mask=decoder_attention_mask,
                )
                logits = outputs.logits
                values = outputs.value
                if hasattr(self.model, "frozen_head") or self.model.peft_type:
                    ref_logits = self.model.forward_hydra(
                        input_ids=prompt_tensors,
                        attention_mask=attention_mask_arg,
                        decoder_input_ids=sample_outputs,
                        decoder_attention_mask=decoder_attention_mask,
                        return_dict=True,
                    ).logits
                else:
                    ref_logits = self.ref_model(
                        input_ids=prompt_tensors,
                        attention_mask=attention_mask_arg,
                        decoder_input_ids=sample_outputs,
                        decoder_attention_mask=decoder_attention_mask,
                        return_dict=True,
                    ).logits
        else:
            all_tokens = torch.cat((prompt_tensors.to(device), sample_outputs), dim=1)
            attention_mask = all_tokens.not_equal(self.tokenizer.pad_token_id).long().to(device)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            batch_size = all_tokens.shape[0]
            with torch.no_grad():
                # TODO: make this output from both aux and primary policy
                attention_mask_arg = attention_mask if batch_size != 1 else None
                logits, *_, values = self.model(
                    all_tokens, attention_mask=attention_mask_arg, position_ids=position_ids
                )
                # TODO(dahoas): When hydra model works need to also support generation on hydra head
                if hasattr(self.model, "frozen_head") or self.model.peft_type:
                    ref_logits = self.model.forward_hydra(
                        all_tokens,
                        attention_mask=attention_mask_arg,
                        position_ids=position_ids,
                        return_dict=True,
                    ).logits
                else:
                    ref_logits = self.ref_model(
                        all_tokens,
                        attention_mask=attention_mask_arg,
                        position_ids=position_ids,
                        return_dict=True,
                    ).logits
                    ref_logits = ref_logits.to(device)
                    
        if self.config.model.model_arch_type == "seq2seq":
            logprobs = logprobs_of_labels(logits[:, :-1, :], sample_outputs[:, 1:])
            ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], sample_outputs[:, 1:])
        else:
            # NOTE: logprob[i] is (log)prob at which all_token[i+1] was sampled
            logprobs = logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:])
            ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], all_tokens[:, 1:])

        n_samples: int = samples.shape[0]

        # Estimate the KL divergence between the model and reference model
        if self.config.model.model_arch_type == "seq2seq":
            attention_mask = sample_outputs != self.tokenizer.pad_token_id
            start = 0
        else:
            start = prompt_tensors.shape[1] - 1

        log_ratio = (logprobs - ref_logprobs) * attention_mask[:, :-1]
        kl = log_ratio.exp() - 1 - log_ratio

        """
        Entropy bonus for exploration
        """
        entropy = -logprobs * attention_mask[:, :-1]

        mean_kl_per_token = kl.mean()
        mean_kl = kl.sum(1).mean()

        logprobs = logprobs.cpu()
        ref_logprobs = ref_logprobs.cpu()
        prompt_tensors = prompt_tensors.cpu()
        sample_outputs = sample_outputs.cpu()
        values = values.cpu()[:, :-1]

        # Get the logprobs and values, for tokens that are not padding,
        # from the end of the prompt up to the <eos> token, while also including the latter
        # (these are taken from the student model and not the reference model)
        ends = start + attention_mask[:, start:].sum(1) + 1
        all_values = [values[ix, start : ends[ix]] for ix in range(n_samples)]
        all_logprobs = [logprobs[ix, start : ends[ix]] for ix in range(n_samples)]

        kl_penalty = self.kl_ctl.value * -log_ratio.cpu()
        kl_penalty = [xs[start : ends[ix]] for ix, xs in enumerate(kl_penalty)]
        
        entropy_bonus = self.config.method.ent_reward_coef * entropy.cpu()
        entropy_bonus = [xs[start : ends[ix]] for ix, xs in enumerate(entropy_bonus)]

        rollout_count = 0

        for sample_idx in range(n_samples):
            sample_length = ends[sample_idx]-start
            if sample_length <= 2:
                rewards = torch.zeros(2)
            else:
                rewards = kl_penalty[sample_idx] + entropy_bonus[sample_idx]
            # Then add in rewards
            if scores.shape[1] == 1:
                # NOTE: Final reward given at EOS token following HHH practice
                rewards[-1] += scores[sample_idx][0].cpu()
            else:
                score = scores[sample_idx]
                score_right_padding = torch.sum(scores_mask[sample_idx])
                score = score[:score_right_padding].cpu()
                p_score = torch.zeros_like(rewards)
                p_score[: score.shape[0]] += score
                rewards += p_score

            ppo_rl_elements.append(
                PPORLElement(
                    query_tensor=prompt_tensors[sample_idx],
                    response_tensor=sample_outputs[sample_idx],
                    logprobs=all_logprobs[sample_idx],
                    values=all_values[sample_idx],
                    rewards=rewards,
                )
            )

            rollout_count += 1
        
        return mean_kl, mean_kl_per_token, rollout_count

    def _aggregate_traj_reward(self, all_scores, all_bleu_scores, all_cossimemb_scores, all_textualsim_scores, all_target_sim_div_scores, all_giberish_scores, device):
        return [
            torch.tensor(score + 
                self.config.method.bleu_reward_coef * bleu_score +
                self.config.method.cossimemb_reward_coef * cossimemb_score +
                self.config.method.textual_sim_reward_coef * textualsim_score + 
                self.config.method.target_sim_div_reward_coef * target_sim_div_score +
                self.config.method.giberish_penalty_coef * giberish_score
                , dtype=torch.float, device=device).view(
                -1,
            )
            for score, bleu_score, cossimemb_score, textualsim_score, target_sim_div_score, giberish_score in zip(
                all_scores, all_bleu_scores, all_cossimemb_scores, all_textualsim_scores, all_target_sim_div_scores, all_giberish_scores)
        ]

    @torch.inference_mode()
    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):  # noqa:
        """Make experiences

        Takes `chunk_size` number of prompts from `prompt_iterator`, samples
        from the model and then computes the KL against a reference model. Finally it
        then appends PPOElements to trainer's `store`.

        Args:
            num_rollouts: Number of rollouts to generate
            iter_count: Total number of updates run (i.e. number of updates run for all batches & epochs)
        """
        logger.info("Collecting rollouts")
        tbar = logging.tqdm(
            total=num_rollouts,
            disable=os.environ.get("RANK", 0) != "0",
            desc=f"[rollout 0 / {num_rollouts}]",
            # Lower progress bar by 1 if we're in WARNING mode or above to avoid hiding high priority progress
            # bars (e.g. loss progress in trainers)
            position=logging.get_verbosity() >= logging.WARNING,
            # Leave progress bar if we're in INFO mode or lower to avoid spamming in suppressed verbosity levels
            leave=logging.get_verbosity() < logging.WARNING,
        )

        clock = Clock()
        ppo_rl_elements = []
        accumulated_stats = []

        while len(ppo_rl_elements) < num_rollouts:
            stats = {}
            # Get next batch in prompt dataset
            batch: PromptBatch = next(self.prompt_iterator)

            rollout_generate_time = time()

            # Generate samples from the language model (similar to using HuggingFace `generate` method)
            attention_mask_arg = batch["attention_mask"] if batch["attention_mask"].shape[0] !=1 else None
            samples = self.generate(batch["input_ids"], attention_mask_arg)
            stats["time/rollout_generate"] = time() - rollout_generate_time

            prompt_tensors = batch.input_ids
            device = samples.device

            prompt_sizes = torch.tensor([prompt_tensors.shape[1]] * len(prompt_tensors), device=device)
            padded_samples = self.accelerator.pad_across_processes(
                samples, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
            )
            padded_prompts = self.accelerator.pad_across_processes(
                prompt_tensors, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
            )
            gathered_samples = self.accelerator.gather(padded_samples)
            gathered_prompts = self.accelerator.gather(padded_prompts)
            gathered_prompt_sizes = self.accelerator.gather(prompt_sizes)
            metadata = gather_dict({k: v for k, v in batch.items() if k != "input_ids" and k != "attention_mask"})
            
            if self.accelerator.is_main_process:
                all_str_samples, all_str_prompts, all_str_outputs = self.decode(
                    gathered_prompts, gathered_samples, gathered_prompt_sizes, append_eos_token=True
                )
                
                rollout_score_time = time()
                # reward_fn should return list of rewards at each token per sample        
                all_scores, all_str_victim_outputs = self.reward_fn(
                            samples=all_str_samples, 
                            prompts=all_str_prompts, 
                            outputs=all_str_outputs,
                            return_texts=True,
                            **metadata)
                """
                Training logs: log all generated texts
                """
                self.train_text_logger.log(
                    all_str_prompts, 
                    all_str_outputs, 
                    all_str_victim_outputs, # TODO: this can be a list of tuples
                    all_scores)

                """
                Compute Self-BLEU rewards as diveristy penalty
                  1. Compute Self-BLEU score for each generated response
                  2. Update the references in Self-BLEU score 
                """
                if self.config.method.bleu_reward_coef == 0:
                    all_bleu_scores = [0.] * len(all_scores)
                else:
                    if self.config.method.bleu_reward_include_prompts:
                        all_bleu_scores = self.bleu_reward_module(all_str_samples)
                        self.bleu_reward_module.append_reference(all_str_samples)
                    else:
                        all_bleu_scores = self.bleu_reward_module(all_str_outputs)
                        self.bleu_reward_module.append_reference(all_str_outputs)

                """
                Compute SimEmd rewards as diversity penalty
                """
                if self.config.method.cossimemb_reward_coef == 0:
                    all_cossimemb_scores = [0.] * len(all_scores)
                else:
                    if self.config.method.cossimemb_reward_include_prompts:
                        all_cossimemb_scores = self.cossimemb_reward_module(all_str_samples)
                        self.cossimemb_reward_module.append_reference(all_str_samples)
                    else:
                        all_cossimemb_scores = self.cossimemb_reward_module(all_str_outputs)
                        self.cossimemb_reward_module.append_reference(all_str_outputs)
                    
                """
                Compute similarity rewards
                """
                if self.config.method.textual_sim_reward_coef == 0:
                    all_textualsim_scores = [0.] * len(all_scores)
                else:
                    if self.config.method.textual_sim_reward_include_prompts:
                        all_textualsim_scores = self.cossimemb_reward_module.compute_similarity(
                            all_str_prompts,
                            all_str_samples)
                    else:
                        all_textualsim_scores = self.cossimemb_reward_module.compute_similarity(
                            all_str_prompts,
                            all_str_outputs)
                        
                """
                Compute target embedding diversity rewards
                """
                if self.config.method.target_sim_div_reward_coef == 0:
                    all_target_sim_div_scores = [0.] * len(all_scores)
                else:                    
                    all_target_sim_div_scores = self.cossimemb_reward_module.compute_l1_div_rewards(
                        all_str_victim_outputs)
                    
                
                """
                Compute gibberish penalty
                """                
                if self.config.method.giberish_penalty_coef == 0:
                    all_giberish_scores = [0.] * len(all_scores)
                else:
                    all_giberish_scores = self.giberish_penalty_penalty_module(all_str_outputs)
                                
                all_scores = self._aggregate_traj_reward(all_scores, all_bleu_scores, all_cossimemb_scores, all_textualsim_scores, all_target_sim_div_scores, all_giberish_scores, device)
                
                # Pad 0 reward on the ends
                all_scores = pad_sequence(all_scores, batch_first=True, padding_value=-np.inf)
                max_len = torch.tensor(all_scores.shape[1], dtype=torch.long, device=device)

                stats["time/rollout_score"] = time() - rollout_score_time

                all_scores = list(all_scores.reshape(self.accelerator.num_processes, -1, max_len).unbind())
                self.history_scores += all_scores
            else:
                all_scores = None
                max_len = torch.tensor(0, dtype=torch.long, device=device)

            if torch.distributed.is_initialized():
                torch.distributed.broadcast(max_len, 0)
                
                scores = torch.empty((len(samples), max_len), device=device)
                torch.distributed.scatter(scores, all_scores)
                
                bleu_scores = torch.empty((len(samples), max_len), device=device)
                torch.distributed.scatter(bleu_scores, all_bleu_scores)
                
                cossimemb_scores = torch.empty((len(samples), max_len), device=device)
                torch.distributed.scatter(cossimemb_scores, all_cossimemb_scores)
                
                textualsim_scores = torch.empty((len(samples), max_len), device=device)
                torch.distributed.scatter(textualsim_scores, all_textualsim_scores)
                
                targetsimdiv_scores = torch.empty((len(samples), max_len), device=device)
                torch.distributed.scatter(targetsimdiv_scores, all_target_sim_div_scores)
                
                giberish_scores = torch.empty((len(samples), max_len), device=device)
                torch.distributed.scatter(giberish_scores, all_giberish_scores)                
            else:
                scores = all_scores[0].clone().detach()
                bleu_scores = torch.tensor(all_bleu_scores).unsqueeze(1).clone().detach().to(scores.device)
                cossimemb_scores = torch.tensor(all_cossimemb_scores).unsqueeze(1).clone().detach().to(scores.device)              
                textualsim_scores = torch.tensor(all_textualsim_scores).unsqueeze(1).clone().detach().to(scores.device)
                targetsimdiv_scores = torch.tensor(all_target_sim_div_scores).unsqueeze(1).clone().detach().to(scores.device)
                giberish_scores = torch.tensor(all_giberish_scores).unsqueeze(1).clone().detach().to(scores.device)
                
            scores_mask = scores != -np.inf

            str_samples, str_prompts, str_outputs = self.decode(prompt_tensors, samples, append_eos_token=True)

            # Pad the sample outputs
            outputs = self.tokenizer(str_outputs).input_ids
            if self.config.model.model_arch_type == "seq2seq":
                # add <pad> to the start of the output
                for i in range(len(outputs)):
                    outputs[i] = [self.tokenizer.pad_token_id] + outputs[i]

            outputs = list(map(torch.LongTensor, outputs))
            maxsize = max(map(len, outputs))
            outputs = [
                F.pad(
                    output,
                    (0, maxsize - len(output)),
                    value=self.tokenizer.pad_token_id,
                )
                for output in outputs
            ]
            sample_outputs = torch.vstack(outputs).to(device)

            if self.config.method.cliprange_reward:
                scores = torch.clip(scores, -self.config.method.cliprange_reward, self.config.method.cliprange_reward)

            # store statistics of the initial rollout as reference
            if self.ref_mean is None:
                self.ref_mean, self.ref_std = (scores * scores_mask).sum(dim=1).mean(), (scores * scores_mask).sum(
                    dim=1
                ).std()
            all_scores_mean, all_scores_std = self.running_moments.update(torch.sum(scores * scores_mask, dim=1))
            stats["rollout_scores/mean"] = all_scores_mean.item()
            stats["rollout_scores/std"] = all_scores_std.item()
            stats["rollout_scores/running_mean"] = self.running_moments.mean.item()
            stats["rollout_scores/running_std"] = self.running_moments.std.item()
            
            stats["rollout_bleu_scores/mean"] = (bleu_scores * scores_mask).mean().item()
            stats["rollout_cossimemb_scores/mean"] = (cossimemb_scores * scores_mask).mean().item()
            stats["rollout_textualsim_scores/mean"] = (textualsim_scores * scores_mask).mean().item()
            stats["rollout_targetsimdiv_scores/mean"] = (targetsimdiv_scores * scores_mask).mean().item()
            stats["rollout_giberish_scores/mean"] = (giberish_scores * scores_mask).mean().item()
            
            if self.config.method.scale_reward == "running":
                scores /= self.running_moments.std
            elif self.config.method.scale_reward == "ref":
                scores /= self.ref_std

            mean_kl, mean_kl_per_token, rollout_count = self._process_element(
                ppo_rl_elements, samples, batch, prompt_tensors, sample_outputs, scores, scores_mask, device)

            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(mean_kl, torch.distributed.ReduceOp.AVG)

            stats["time/rollout_time"] = clock.tick()
            stats["policy/sqrt_kl"] = torch.sqrt(mean_kl).item()
            stats["policy/kl_per_token"] = torch.sqrt(mean_kl_per_token).item()
            accumulated_stats.append(stats)

            tbar.set_description(f"[rollout {len(ppo_rl_elements)} / {num_rollouts}]")
            tbar.update(min(rollout_count, num_rollouts))
        tbar.close()

        stats = {k: sum([xs[k] for xs in accumulated_stats]) / len(accumulated_stats) for k in stats}
        stats["kl_ctl_value"] = self.kl_ctl.value
        self.mean_kl = stats["policy/sqrt_kl"] ** 2
        self.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to trainer's rollout storage
        self.push_to_store(ppo_rl_elements)

    
    def evaluate(self):  # noqa: C901
        """Samples model on `eval_prompts`, logs stats with `reward_fn` or `metric_fn` if provided"""
        logger.info("Evaluating model")

        # Do multiple evaluations over a single list in `gen_kwargs` if present
        if self.generate_sweep_kwarg is not None:
            gen_sweep_arg, gen_sweep_values = self.generate_sweep_kwarg
        else:
            gen_sweep_values = [None]

        desc = [
            f"generation sweep 0/{len(gen_sweep_values)}",
            f"eval batch 0/{len(self.eval_dataloader)}",
        ]
        tbar = logging.tqdm(
            total=len(self.eval_dataloader) * len(gen_sweep_values),
            desc=f"[{' | '.join(desc)}]",
            disable=not self.accelerator.is_main_process,
            position=0,
            leave=True,
        )

        stats = {}
        table = []

        for i_sweep, gen_sweep_value in enumerate(gen_sweep_values):
            # A dedicated suffix for wandb logging
            if gen_sweep_value is not None:
                sweep_suffix = f"@{gen_sweep_arg}={gen_sweep_value}"
            else:
                sweep_suffix = ""

            all_samples = []
            all_prompts = []
            all_prompt_sizes = []
            all_metadata = []
            generate_time = time()
            for i_prompt, prompts in enumerate(self.eval_dataloader):
                metadata = {k: v for k, v in prompts.items() if k != "input_ids" and k != "attention_mask"}
                if self.generate_sweep_kwarg:
                    samples = self.generate_eval(
                        prompts["input_ids"], prompts["attention_mask"], **{gen_sweep_arg: gen_sweep_value}
                    )
                else:
                    samples = self.generate_eval(prompts["input_ids"], prompts["attention_mask"])

                # TODO(reciprocated): this should be moved into `decode`
                # but that needs to be synced with indexing in `make_experience`
                if self.config.model.model_arch_type == "seq2seq":
                    samples = samples[:, 1:].contiguous()

                prompt_sizes = torch.tensor(prompts.input_ids.shape[1]).repeat(len(prompts.input_ids))
                prompts, samples, prompt_sizes = self.accelerator.gather_for_metrics(
                    self.accelerator.pad_across_processes(
                        [prompts.input_ids, samples, prompt_sizes.to(samples.device)],
                        dim=1,
                        pad_index=self.tokenizer.pad_token_id,
                    )
                )
                all_samples.extend(samples.tolist())
                all_prompts.extend(prompts.tolist())
                all_prompt_sizes.extend(prompt_sizes.tolist())

                metadata = gather_dict(metadata, self.accelerator.gradient_state)
                all_metadata.append(metadata)

                desc = [
                    f"generation sweep {i_sweep + 1}/{len(gen_sweep_values)}",
                    f"eval batch {i_prompt + 1}/{len(self.eval_dataloader)}",
                ]
                tbar.set_description(f"[{' | '.join(desc)}]")
                tbar.update()
            tbar.close()

            stats["time/generate"] = time() - generate_time

            if self.accelerator.is_main_process:
                str_all_samples, str_all_prompts, str_all_outputs = self.decode(all_prompts, all_samples, all_prompt_sizes)
                
                # NOTE: make batch otherwise cannot evaluate on a large test set
                eval_batch_size = self.config.train.batch_size
                for eval_batch_i in range(int(np.floor(len(str_all_samples) / eval_batch_size)) + 1):
                    str_samples = str_all_samples[eval_batch_i*eval_batch_size:(eval_batch_i+1)*eval_batch_size]
                    str_prompts = str_all_prompts[eval_batch_i*eval_batch_size:(eval_batch_i+1)*eval_batch_size]
                    str_outputs = str_all_outputs[eval_batch_i*eval_batch_size:(eval_batch_i+1)*eval_batch_size]

                    if hasattr(self.config.model, "human_attacker_template_pool") and self.config.model.human_attacker_template_pool is not None:
                        attacker_template_pool = self.config.model.human_attacker_template_pool.split("\n")
                        attacker_template_pool = [v for v in attacker_template_pool if len(v) > 0]
                        attacker_template_samples = np.random.choice(attacker_template_pool, len(str_outputs)).tolist()
                        str_outputs_new = []
                        for v1, v2 in zip(str_prompts, attacker_template_samples):
                            if "<CONTEXT>" in v2:
                                str_out = v2.replace("<CONTEXT>", v1)
                            else:
                                str_out = v2
                            str_outputs_new.append(str_out)
                        str_outputs = str_outputs_new

                    if eval_batch_i == 0:
                        columns = ["attacker prompt", "attacker output (victim prompt)"]
                    columns_data = [str_prompts, str_outputs]

                    metadata, *xs = all_metadata
                    for k in metadata:
                        for x in xs:
                            metadata[k].extend(x[k])

                    # in online setting, compute the reward for validation
                    if self.reward_fn:
                        logger.info("Computing rewards")                   
                        rewards, victim_str_outputs = self.reward_fn(
                                samples=str_samples, 
                                prompts=str_prompts, 
                                outputs=str_outputs,
                                return_texts=True,
                                **metadata)

                        self.eval_text_logger.log( # TODO: not sure why the iter_count gets reset at every eval batch
                            str_prompts,
                            str_outputs,
                            victim_str_outputs,
                            rewards,
                        )
                        
                        rewards = torch.tensor(
                            rewards,
                            dtype=float,
                        )

                        if eval_batch_i == 0:
                            columns.append("victim output")
                        columns_data.append([victim_str_output \
                                            for str_prompt, str_output, victim_str_output in 
                                                zip(str_prompts, str_outputs, victim_str_outputs)])
                    
                        mean_reward = rewards.mean().item()
                        if eval_batch_i == 0:
                            columns.append("reward")
                        if not isinstance(rewards, list):
                            rewards = rewards.tolist()
                        columns_data.append(rewards)
                        stats[f"reward/mean{sweep_suffix}"] = mean_reward # TODO: only get the last one
                        # stats[f"reward/train/mean/{sweep_suffix}"] = np.mean(self.history_scores)

                    # additionally log any other metrics
                    if self.metric_fn:
                        logger.info("Computing metrics")
                        metric_time = time()
                        metrics = self.metric_fn(samples=str_samples, prompts=str_prompts, outputs=str_outputs, **metadata)
                        stats["time/metric"] = time() - metric_time

                        mean_metrics = {
                            f"metrics/{k}{sweep_suffix}": torch.as_tensor(xs).mean(-1).item() for k, xs in metrics.items()
                        }

                        stats.update(mean_metrics)

                        for metric, values in metrics.items():
                            # Skip metrics that are scalers since they represent aggregated values
                            if isinstance(values, float):
                                continue
                            columns.append(metric)
                            if not isinstance(values, list):
                                values = values.tolist()
                            columns_data.append(values)

                    # Prepend the sweep argument along with samples
                    if self.generate_sweep_kwarg:
                        columns.insert(0, gen_sweep_arg)
                        columns_data.insert(0, [gen_sweep_value] * len(samples))

                    table.append(list(zip(*columns_data)))

        # Log and display evaluation metrics
        logger.info("Summarizing evaluation")
        if self.accelerator.is_main_process:
            rows = sum(list(map(list, zip(*table))), [])

            # Add metrics/rewards to the table's title
            table_title = f"Evaluation #{self.nth_evaluation}"
            for k, x in stats.items():
                if k.startswith("reward") or k.startswith("metrics"):
                    table_title += f" {k}: {significant(x)}"

            rich_table = Table(*columns, title=table_title, show_lines=True)
            for ix in range(max(min(3, len(rows)), len(gen_sweep_values))):
                rich_table.add_row(*[str(significant(x)) for x in rows[ix]])
            Console().print(rich_table)

            if self.config.train.tracker == "wandb":
                import wandb

                stats["samples"] = wandb.Table(columns, rows)

        self.nth_evaluation += 1
        return stats