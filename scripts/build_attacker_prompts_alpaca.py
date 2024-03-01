from pathos.multiprocessing import ProcessingPool as Pool

import jsonlines
import random
from tqdm import *
from datasets import load_dataset

alpaca_dataset = load_dataset("tatsu-lab/alpaca", 
                    split="train").filter(lambda example: len(example["input"]) == 0)

n_prompts = 1024
n_examples_per_prompt = 3

def _make_prompt(_):
    prompt = "Write a list of instructions:\n"
    for j, example in enumerate(random.choices(
            alpaca_dataset["instruction"], 
            k=n_examples_per_prompt)):
        prompt += f"{j + 1}. {example}\n"
    prompt += f"{n_examples_per_prompt + 1}."
    return prompt

with jsonlines.open("prompts/alpaca.jsonl", "w") as writer:
    with Pool() as p:
        with tqdm(total=n_prompts) as pbar:
            for prompt in p.imap(_make_prompt, range(n_prompts)):
                pbar.update()
                writer.write({"attacker_prompt": prompt})

with jsonlines.open("prompts/alpaca.jsonl", "r") as reader:
    for line in reader:
        print(line)


