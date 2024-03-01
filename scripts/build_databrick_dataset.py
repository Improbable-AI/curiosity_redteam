from pathos.multiprocessing import ProcessingPool as Pool

import jsonlines
import random
from tqdm import *
from datasets import load_dataset

dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

n_prompts = 10000
n_examples_per_prompt = 2

def _make_prompt(_):
    prompt = "Ask questions:\n"
    for j, example in enumerate(random.choices(
            dataset["instruction"], 
            k=n_examples_per_prompt)):
        prompt += f"{j + 1}. {example}\n"
    prompt += f"{n_examples_per_prompt + 1}."
    return prompt

with jsonlines.open(f"prompts/databrick-{n_prompts}-{n_examples_per_prompt}.jsonl", "w") as writer:
    with Pool() as p:
        with tqdm(total=n_prompts) as pbar:
            for prompt in p.imap(_make_prompt, range(n_prompts)):
                pbar.update()
                writer.write({"attacker_prompt": prompt})

with jsonlines.open(f"prompts/databrick-{n_prompts}-{n_examples_per_prompt}.jsonl", "r") as reader:
    for line in reader:
        print(line)


