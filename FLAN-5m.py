import random

# System Messages
# Page 9, Table 2
SM = {
    1: "",
    2: "You are an AI assistant. Provide a detailed answer so user don’t need to search outside to understand the answer.",
    3: "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.",
    4: "You are a helpful assistant, who always provide explanation. Think like you are answering to a five year old.",
    5: "You are an AI assistant that follows instruction extremely well. Help as much as you can.",
    6: "You are an AI assistant that helps people find information. Provide a detailed answer so user don’t need to search outside to understand the answer.",
    7: "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.",
    8: "You should describe the task and explain your answer. While answering a multiple choice question, first output the correct answer(s). Then explain why other answers are wrong. Think like you are answering to a five year old.",
    9: "Explain how you used the definition to come up with the answer.",
    10: "You are an AI assistant. You should describe the task and explain your answer. While answering a multiple choice question, first output the correct answer(s). Then explain why other answers are wrong. You might need to use additional knowledge to answer the question.",
    11: "You are an AI assistant that helps people find information. User will you give you a question. Your task is to answer as faithfully as you can. While answering think step-bystep and justify your answer.",
    12: "User will you give you a task with some instruction. Your job is follow the instructions as faithfully as you can. While answering think step-by-step and justify your answer.",
    13: "You are a teacher. Given a task, you explain in simple steps what the task is asking, any guidelines it provides and how to use those guidelines to find the answer.",
    14: "You are an AI assistant, who knows every language and how to translate one language to another. Given a task, you explain in simple steps what the task is asking, any guidelines that it provides. You solve the task and show how you used the guidelines to solve the task.",
    15: "Given a definition of a task and a sample input, break the definition into small parts.\nEach of those parts will have some instruction. Explain their meaning by showing an example that meets the criteria in the instruction. Use the following format:\nPart  # : a key part of the definition.\nUsage: Sample response that meets the criteria from the key part. Explain why you think it meets the criteria.",
    16: "You are an AI assistant that helps people find information.",
}

# System Message Pickers 
# Figure 6 page 10
sm_cot = lambda: SM[random.choice([6, 11, 16])]
sm_niv2 = lambda: SM[random.choice([1, 2, 5, 7, 9, 12, 13, 14, 15])]
sm_t0 = lambda: SM[random.choice([1, 2, 3, 5, 7])]
sm_flan2021 = lambda multiple_choice: SM[random.choice([8, 10])] if multiple_choice else SM[random.choice([3, 4, 7])]

def download_dataset(dataset_name):
    if dataset_name.lower() == "cot":
        cot = iter(datasets.load_dataset("conceptofmind/cot_submix_original", streaming=True))
        process_cot(cot)
    elif dataset_name.lower() == "niv":
        niv = iter(datasets.load_dataset("conceptofmind/niv2_submix_original", streaming=True))
        process_niv(niv)
    elif dataset_name.lower() == "flan":
        flan = iter(datasets.load_dataset("conceptofmind/flan2021_submix_original", streaming=True))
        process_flan(flan)
    elif dataset_name.lower() == "t0":
        t0 = iter(datasets.load_dataset("conceptofmind/t0_submix_original", split="train", streaming=True))
        process_t0(t0)

import os
import json
import pandas as pd
from IPython.display import display
import datasets
import tqdm
from check_if_multiple_choice import check_if_multiple_choice

# Table 3 Page 10
cot_total = 150000
niv_total = 440000
flan_total = 2500000
t0_total = 2000000

output_dir = "COT"
os.makedirs(output_dir, exist_ok=True)
cot = iter(datasets.load_dataset(
    "conceptofmind/cot_submix_original", streaming=True))

def process_cot(cot):
    with open("data/cot.jsonl", "w") as f:
        stream = tqdm.tqdm(cot, total=cot_total)
        
        for i, data in enumerate(stream):
             if data['template_type'] not in ['zs_opt', 'zs_noopt']:
                continue
            
            question = data['inputs']
            system_prompt = sm_cot()
            json.dump({"id": f"cot.{i}", "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]}, f)
            f.write("\n")
            
            stream.update(i)
            if i >= cot_total:
                break

niv = iter(datasets.load_dataset(
    "conceptofmind/niv2_submix_original", streaming=True))


def process_niv(niv):
    with open("data/niv.jsonl", "w") as f:
        stream = tqdm.tqdm(niv, total=niv_total)
        task_counts = {}
        
        for i, data in enumerate(stream):
            task_id = data['task_id']
            task_counts.setdefault(task_id, 0)
            
            if task_counts[task_id] < 300:
                question = data['inputs']
                system_prompt = sm_niv2()
                json.dump({"id": f"niv.{i}", "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]}, f)
                f.write("\n")
                
                task_counts[task_id] += 1
                
            stream.update(i)
            if i >= niv_total:
                break
flan = iter(datasets.load_dataset(
    "conceptofmind/flan2021_submix_original", streaming=True))

def sample_queries(tasks, n, max_queries_per_task=1000000):
    Q = []
    while len(Q) < n:
        t = random.choice(tasks)
        if not t or len(t) >= max_queries_per_task:
            tasks.remove(t)
            continue
        q = random.choice(t)
        t.remove(q)
        Q.append(q)
    return Q

def process_flan(flan):
    tasks = {}
    for data in flan:
        task_id = data['task_id']
        if task_id not in tasks:
            tasks[task_id] = []
        tasks[task_id].append(data)

    sampled_queries = sample_queries(list(tasks.values()), flan_total)

    with open("data/flan.jsonl", "w") as f:
        stream = tqdm.tqdm(sampled_queries, total=flan_total)
        
        for i, data in enumerate(stream):
            question = data['inputs']
            multiple_choice = check_if_multiple_choice(question)
            system_prompt = sm_flan2021(multiple_choice)
            json.dump({"id": f"flan.{i}", "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]}, f)
            f.write("\n")
            
            stream.update(i)
            if i >= flan_total:
                break

t0_total = 2000000

T0 = iter(datasets.load_dataset(
    "conceptofmind/t0_submix_original", split=train, streaming=True))

def process_t0(t0):
    tasks = {}  
    for data in t0:
        task_id = data['task_id']
        if "big-bench" not in task_id.lower():
            if task_id not in tasks:
                tasks[task_id] = []
            tasks[task_id].append(data)

    sampled_queries = sample_queries(list(tasks.values()), t0_total)

    with open("data/t0.jsonl", "w") as f:
        stream = tqdm.tqdm(sampled_queries, total=t0_total)
        
        for i, data in enumerate(stream):
            question = data['inputs']
            system_prompt = sm_t0()
            json.dump({"id": f"t0.{i}", "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]}, f)
            f.write("\n")
            
            stream.update(i)
            if i >= t0_total:
                break

print("Please choose a dataset to download and process:")
print("1. COT")
print("2. NIV")
print("3. FLAN")
print("4. T0")

dataset_options = {
    "1": "cot",
    "2": "niv",
    "3": "flan",
    "4": "t0",
}

selected_option = input("Enter the corresponding digit: ")

download_dataset(dataset_options[selected_option])