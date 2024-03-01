import os
import sys
import copy
import itertools
import datetime
import random
import json
from functools import reduce
from pathlib import Path

from experiments import get_launch_args, sweep, sweep_with_devices, launch, exp_id, parse_task_ids, launch_jobs, get_local_storage_path

import multiprocessing

def maybe_escape_quote(s, mode):
  if mode in ["screen", "sbatch"]:
    return json.dumps(s)[1:-1]
  else:
    return s

if __name__ == '__main__':
  experiment = f"{os.path.basename(os.path.dirname(Path(__file__)))}"
  launch_args = get_launch_args(experiment)

  algos_scripts = [
    {"script": "CUDA_VISIBLE_DEVICES={gpu} python3 ppo_gpt2_gpt2_imdb_toxicity_response.py",},
  ]

  init_kl_coefs = [
    # 1.0,
    # 0.1,
    # 0.01,
    0.001,
    # 0.0,
  ]

  bleu_reward_coefs = [
    -1.0,
    # -0.5,
    # -0.1,
    # -0.01,
    # 0,
  ]
  
  bleu_tokenizers = [
    "nltk",
    # "model",
  ]

  cossimemb_reward_coefs = [
    -1.0,
    # -0.5,
    # -0.1,
    # -0.01,
    # 0,
  ]

  ent_reward_coefs = [
    # 1.0,
    # 0.1,
    0.01,
    # 0.001,
    # 0.0
  ]
  
  textual_sim_reward_coefs = [
    # 1.0,
    0,
  ]
  
  target_sim_div_reward_coefs = [
    0.0,
    # 1.0,
  ]
  
  batch_sizes = [
    64,
    # 32,
  ]
  
  seeds = [
    1000,
    2000,
    3000,
  ]

  all_job_args = []
  for job_idx, (n_tasks, device,           
          algo_script,
          init_kl_coef,          
          bleu_reward_coef,
          bleu_tokenizer,
          cossimemb_reward_coef,
          ent_reward_coef,
          textual_sim_reward_coef,
          target_sim_div_reward_coef,
          batch_size,
          seed) in enumerate(
          sweep_with_devices(itertools.product(           
            algos_scripts,
            init_kl_coefs,
            bleu_reward_coefs,
            bleu_tokenizers,
            cossimemb_reward_coefs,
            ent_reward_coefs,
            textual_sim_reward_coefs,
            target_sim_div_reward_coefs,
            batch_sizes,
            seeds),
            devices=launch_args.gpus,
            n_jobs=launch_args.n_jobs,
            n_parallel_task=1, shuffle=True)):
    job_args = []
    for task_idx in range(n_tasks):
      args = [
        algo_script[task_idx]["script"].format(gpu=device),
      ]
      
      if launch_args.debug:
        base_dir = "debug"
        suffix = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
      else:
        base_dir = "results"
        suffix = seed[task_idx]

      logdir = f"{base_dir}/imdb_toxicity_response/ppo{batch_size[task_idx]}_kl{init_kl_coef[task_idx]}_bleu{bleu_reward_coef[task_idx]}_bleuToken{bleu_tokenizer[task_idx]}_cossimemb{cossimemb_reward_coef[task_idx]}_ent{ent_reward_coef[task_idx]}_targdiv{target_sim_div_reward_coef[task_idx]}/{suffix}"
      config = maybe_escape_quote(json.dumps({
          "method.init_kl_coef": init_kl_coef[task_idx],
          "method.bleu_reward_coef": bleu_reward_coef[task_idx],
          "method.cossimemb_reward_coef": cossimemb_reward_coef[task_idx],
          "method.ent_reward_coef": ent_reward_coef[task_idx],
          "method.bleu_reward_grams": "[2, 3, 4, 5]",
          "method.bleu_tokenizer": bleu_tokenizer[task_idx],
          "method.target_sim_div_reward_coef": target_sim_div_reward_coef[task_idx],
          "train.logging_dir": logdir,
          "train.batch_size": batch_size[task_idx],
          "train.checkpoint_dir": logdir,
          "train.seed": seed[task_idx],
        }), mode=launch_args.mode)
      args.append(f"'{config}'")
      job_args.append(" ".join(args))
    all_job_args.append(job_args[0])

    if launch_args.debug:
      break

  print(f"Total: {len(all_job_args)}")

  launch_jobs(experiment,
    all_job_args,
    *parse_task_ids(launch_args.task_id),
    n_jobs=launch_args.n_jobs,
    mode=launch_args.mode,
    script="")

  print(f"Total: {len(all_job_args)}, num_gpus={len(launch_args.gpus)}")