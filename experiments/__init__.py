import os
import math
import itertools
import random


BASE_LOCAL_SCRIPT = """{entry_script} {args}"""

def exp_id():
  import uuid
  return uuid.uuid4()

def hostname():
  import subprocess
  cmd = 'hostname -f'
  try:
      p = subprocess.check_output(cmd, shell=True)  # Save git diff to experiment directory
      return p.decode('utf-8').strip()
  except subprocess.CalledProcessError as e:
      print(f"can not get obtain hostname via `{cmd}` due to exception: {e}")
      return None

def get_local_storage_path():
  return "."

def local(job_name, args, entry_script, verbose=False, **kwargs):
    assert len(args) == 1 # Not support parallel screen jobs for now
    script = BASE_LOCAL_SCRIPT.format(args=args[0], entry_script=entry_script)
    cmd = f"{script}"

    if verbose:
      print(cmd)
    os.system(cmd)

def launch(job_name, args, mode, entry_script, verbose=False, **kwargs):
  if mode == 'local':
    local(job_name, args, entry_script, verbose=verbose, **kwargs)
  else:
    raise NotImplemented()

def parse_task_ids(task_ids):
  if task_ids == ":":
    start_task_id = 0
    end_task_id = None
  else:
    start_task_id, end_task_id = map(lambda x: int(x), task_ids.split(":"))

  return start_task_id, end_task_id

def launch_jobs(experiment, all_job_args, start_task_id, end_task_id, n_jobs, mode, script):
  import time, math
  if end_task_id is None:
    end_task_id = len(all_job_args) - 1
  job_launched = 0
  job_size = math.ceil((end_task_id + 1 - start_task_id) / n_jobs) if n_jobs >= 1 else 1
  expID = exp_id()
  for idx, i in enumerate(range(start_task_id, end_task_id + 1, job_size)):
    launch(f"{experiment}_{expID}_{idx}", all_job_args[i: min(i + job_size, end_task_id + 1)],
        mode=mode, entry_script=script, expID=expID,
        verbose=True)
    print(f"Run task {i}-{min(i + job_size, end_task_id + 1)}")
    job_launched += 1
  print(f"Launched {job_launched} jobs. Each job runs {job_size} tasks.")


def get_launch_args(experiment):
  import argparse
  parser = argparse.ArgumentParser(description=f'{experiment}')
  parser.add_argument('--gpus', nargs="+", type=str, help="GPU ID lists (e.g., 0,1,2,3)", default="0")
  parser.add_argument('--mode', type=str, choices=['sbatch', 'sbatch-cpu', 'screen', 'bash', 'local', "gcp", "docker"], required=True)
  parser.add_argument('--n_parallel_task', type=int, default=1, help="Number of parallel jobs in on sbatch submission")
  parser.add_argument('--task_id', help="e.g., 5:10", type=str, required=False, default=":")
  parser.add_argument('--debug', action='store_true', default=False)
  parser.add_argument('--seq', action='store_true', default=False)
  parser.add_argument('--n_jobs', type=int, help="Number of jobs running in sequence.", default=-1)
  parser.add_argument('--n_task_per_gpu', type=int, help="Number of tasks running on the same gpu.", default=1)
  parser.add_argument('--tags', nargs="+", type=str, default=[])
  parser.add_argument('--run_name', type=str, default=None)
  args = parser.parse_args()
  args.gpus = [args.gpus] if isinstance(args.gpus, int) else args.gpus
  return args

def to_tuple_list(list_tuple):
  tuple_lists = [[] for i in range(len(list_tuple[0]))]
  for t in list_tuple:
    for i, e in enumerate(t):
      tuple_lists[i].append(e)
  return tuple_lists

def sweep(sweep_args, n_parallel_task=1, shuffle=False):
  buffer = [] # a list of tuple, each tuple is one arg combination
  if shuffle:
    sweep_args = list(sweep_args)
    random.shuffle(sweep_args)
  n_args = len(sweep_args)
  for args in sweep_args:
    buffer.append(args)
    if len(buffer) == n_parallel_task:
      yield (len(buffer), *to_tuple_list(buffer))
      buffer = []
  if len(buffer) > 0:
    yield (len(buffer), *to_tuple_list(buffer))
    buffer = []

def sweep_with_devices(sweep_args, devices, n_jobs, n_parallel_task=1, shuffle=False):
  buffer = [] # a list of tuple, each tuple is one arg combination
  sweep_args = list(sweep_args)
  if shuffle:    
    random.shuffle(sweep_args)

  n_args = len(sweep_args)
  n_tasks_per_device = math.ceil(n_args / n_jobs)

  for idx, args in enumerate(sweep_args):
    buffer.append(args)
    device = devices[(idx // n_tasks_per_device) % len(devices)]
    if len(buffer) == n_parallel_task:
      yield (len(buffer), device, *to_tuple_list(buffer))
      buffer = []

  if len(buffer) > 0:
    device = devices[(idx // n_tasks_per_device) % len(devices)]
    yield (len(buffer), device, *to_tuple_list(buffer))
    buffer = []