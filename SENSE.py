#This script is for SENSE prompting, currently set to aya. model paths commented, output names include model names
import re
import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json


model_id = "CohereLabs/aya-23-8B"
#model_id = "google/gemma-3-4b-it"
#model_id = "google/gemma-3-12b-it"
#model_id = "google/gemma-3-27b-it"
#model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#model_id = "mistralai/Mistral-7B-Instruct-v0.3"
#model_id = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

##### DATA PRE-PROCESSING #####
def load_data(target_language: str):
    prefix = target_language.lower()[:4]
    base_path = f'/home/common/ACNLP/umr_applications/lpp/{prefix}_experiment_data'
    
    with open(f'{base_path}/eng_sent.txt', 'r', encoding='utf-8') as f:
        eng_lines = f.readlines()
    with open(f'{base_path}/{prefix[:3]}_sent.txt', 'r', encoding='utf-8') as f:
        tgt_lines = f.readlines()
    
    return eng_lines, tgt_lines

def build_prompt(examples, n_shots, target_language):
    if n_shots > 0:
        prompt = f"Given the following examples, please translate the final English sentence into {target_language} by utilizing its semantic parsing result which helps to understand grammar and semantics (output ONLY the translation): \n"
        
        for eng, tgt in examples:
            prompt += f"English: {eng.strip()}\n"
            prompt += f" {target_language}: {tgt}\n"
        
    else:
        prompt = f"Please translate this English sentence into {target_language} by utilizing its semantic parsing result which helps to understand grammar and semantics (output ONLY the translation): \n"
    
    return prompt

def translate(messages, target_language: str):
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(inputs, max_new_tokens=128)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")[-1]
    lines = response.splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    return " ".join(lines) if lines else ""

def run(output_file, n_shots, eng_lines, tgt_lines, target_language, directory):
    prefix = target_language.lower()[:4]
    with open(f'/home/common/ACNLP/umr_applications/lpp/{prefix}_experiment_data/five_shot.json', 'r') as f:
        all_indices = json.load(f)
    
    with open(f'{directory}/{output_file}', 'w', encoding='utf-8') as out:
        for i in tqdm(range(len(eng_lines))):
            indices = all_indices[i][5-n_shots:]
           
            examples = [(eng_lines[idx].strip(), tgt_lines[idx]) for idx in indices]
           
            prompt = build_prompt(examples, n_shots, target_language)
           
            content = prompt + f"English: {eng_lines[i].strip()}\n"
            content += f"{target_language}: "

            messages = [
                {"role": "system", "content": "You are a helpful translation assistant."},
                {"role": "user", "content": content}
            ]
            
            translation = translate(messages, target_language)
            if i < len(eng_lines) - 1:
                out.write(translation + "\n")
            else:
                out.write(translation)

##### MAIN #####
def main(target_language: str):
    print (target_language)
    directory = f'/home/common/ACNLP/umr_applications/ellie_experiments/aya/{target_language}'
    os.makedirs(directory, exist_ok=True)
    
    eng_lines, tgt_lines = load_data(target_language)

    shot_configs = [0, 1, 3, 5]
    
    for n_shots in shot_configs:
        run(f'aya_{n_shots}_shot_sense', n_shots, eng_lines, tgt_lines, target_language, directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('language', type=str)
    
    args = parser.parse_args()
    target_language = args.language
    main(target_language)