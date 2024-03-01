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
  experiment = f"{os.path.basename(os.path.dirname(Path(__file__)))}_gpt2"
  launch_args = get_launch_args(experiment)

  algos_scripts = [
    {"script": "CUDA_VISIBLE_DEVICES={gpus} python3 ppo_gpt2_dolly_peft_databrick_toxicity.py",},
    # {"script": "CUDA_VISIBLE_DEVICES={gpu1},{gpu2} python3 redteam_instruction_dolly.py",},
  ]

  init_kl_coefs = [
    # 1.0,
    # 0.1,
    # 0.01,
    0.001,
    # 0.0001,
  ]

  bleu_reward_coefs = [
    # -1.0,
    # -0.5,
    # -0.1,
    # -0.01,
    0,
  ]

  cossimemb_reward_coefs = [
    # -1.0,
    # -0.5,
    # -0.1,
    # -0.01,
    0,
  ]

  ent_reward_coefs = [
    # 1.0,
    # 0.1,
    # 0.01,
    # 0.001,
    0.0
  ]
  
  textual_sim_reward_coefs = [
    0,
  ]
  
  giberish_penalty_coefs = [
    1.0,
    # 0.1,
    # 0.0,
  ]
  
  target_sim_div_reward_coefs = [
    0.0,
    # 1.0,
  ]
  
  batch_sizes = [
    # 32,
    64,
    # 256,
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
          cossimemb_reward_coef,
          ent_reward_coef,
          textual_sim_reward_coef,
          giberish_penalty_coef,
          target_sim_div_reward_coef,
          batch_size,
          seed) in enumerate(
          sweep_with_devices(itertools.product(           
            algos_scripts,
            init_kl_coefs,
            bleu_reward_coefs,
            cossimemb_reward_coefs,
            ent_reward_coefs,
            textual_sim_reward_coefs,
            giberish_penalty_coefs,
            target_sim_div_reward_coefs,
            batch_sizes,
            seeds),
            devices=launch_args.gpus,
            n_jobs=launch_args.n_jobs,
            n_parallel_task=1, shuffle=True)):
    job_args = []
    for task_idx in range(n_tasks):
      args = [
        algo_script[task_idx]["script"].format(gpus=device),
      ]
      
      if launch_args.debug:
        suffix = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        base_dir = "debug"
      else:
        suffix = seed[task_idx]
        base_dir = "results"

      logdir = f"{base_dir}/databricks_toxicity_gpt2_dolly_peft/ppo{batch_size[task_idx]}_gpt2_kl{init_kl_coef[task_idx]}_bleu{bleu_reward_coef[task_idx]}_cossimemb{cossimemb_reward_coef[task_idx]}_ent{ent_reward_coef[task_idx]}_textsim{textual_sim_reward_coef[task_idx]}_gebrish{giberish_penalty_coef[task_idx]}_targdiv{target_sim_div_reward_coef[task_idx]}/{suffix}"
      config = maybe_escape_quote(json.dumps({
          "method.init_kl_coef": init_kl_coef[task_idx],
          "method.bleu_reward_coef": bleu_reward_coef[task_idx],
          "method.cossimemb_reward_coef": cossimemb_reward_coef[task_idx],
          "method.ent_reward_coef": ent_reward_coef[task_idx],
          "method.bleu_reward_grams": "[2, 3, 4, 5]",
          "method.textual_sim_reward_coef": textual_sim_reward_coef[task_idx],
          "method.target_sim_div_reward_coef": target_sim_div_reward_coef[task_idx],
          "method.giberish_penalty_coef": giberish_penalty_coef[task_idx],
          "method.giberish_model_device": "cpu", # HACK: we can change to cuda:1
          "method.cossimemb_model_device": "cpu", # HACK: we can change to cuda:1
          "train.batch_size": batch_size[task_idx],
          "train.logging_dir": logdir,
          "train.checkpoint_dir": logdir,
          "train.seed": seed[task_idx],
          "train.minibatch_size": 8, # 64,
          "method.chunk_size": 8, # 64,
          # "method.reward_model_device_offset": 1,
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