import re
import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json

model_id = "CohereLabs/aya-23-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

##### OUTPUT CLEANING #####
def extract_translated_text(response: str, target_language: str) -> str:
    response = response.split(f"{target_language}:<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>", 1)[1].strip()
    lines = response.splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    return " ".join(lines) if lines else ""

##### DATA PRE-PROCESSING #####
def load_data(target_language: str):
    if target_language == "Mandarin":
        prefix = "chin"
    else:
        prefix = target_language.lower()[:4]
    
    base_path = f'umr-amr-mt-prompting/lpp/{prefix}_experiment_data'
    
    with open(f'{base_path}/eng_sent.txt', 'r', encoding='utf-8') as f:
        eng_lines = f.readlines()
    with open(f'{base_path}/eng_umrs.txt', 'r', encoding='utf-8') as f:
        umr_lines = f.readlines()
    with open(f'{base_path}/eng_amrs.txt', 'r', encoding='utf-8') as f:
        amr_lines = f.readlines()
    with open(f'{base_path}/{prefix[:3]}_sent.txt', 'r', encoding='utf-8') as f:
        tgt_lines = f.readlines()
    
    return eng_lines, umr_lines, amr_lines, tgt_lines

def build_prompt(examples, representation_type, n_shots, target_language):
    if n_shots > 0:
        prompt = "Given the following examples of English sentences"
        
        if representation_type:
            prompt += f" (accompanied by {representation_type} parses)"
        
        prompt += f" and their {target_language} translations:\n"
    else:
        prompt = ""
    
    for eng, xmr, tgt in examples:
        prompt += f"English: {eng.strip()}"
        if representation_type and xmr:
            prompt += f" {representation_type}: {xmr.strip()}"
        prompt += f" {target_language}: {tgt}"
    
    prompt += f"\nTask: Meaning-Preserving Translation"
    if representation_type:
        prompt += f" Using {representation_type}"
    prompt += "\n\n"
    
    prompt += "Instructions:\n"
    if representation_type:
        prompt += f"- Use the {representation_type} as the full semantic structure.\n"
        prompt += "- Preserve predicate–argument relations.\n"
        prompt += "- Preserve negation, modifiers, quantities, and named entities.\n"
    prompt += f"- Please output ONLY the translation.\n\n"
    
    prompt += f"Translate the following English sentence into {target_language}:\n"

    return prompt

def translate(messages, target_language: str):
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(inputs, max_new_tokens=128)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)
    return extract_translated_text(response, target_language)

def run(output_file, n_shots, eng_lines, xmr_lines, tgt_lines, representation_type, target_language, directory):
    if target_language == "Mandarin":
        prefix = "chin"
    else:
        prefix = target_language.lower()[:4]

    base_path = f'umr-amr-mt-prompting/lpp/{prefix}_experiment_data'
    with open(f'umr-amr-mt-prompting/lpp/{prefix}_experiment_data/five_shot.json', 'r') as f:
        all_indices = json.load(f)
    
    with open(f'{directory}/{output_file}', 'w', encoding='utf-8') as out:
        for i in tqdm(range(len(eng_lines))):
            indices = all_indices[i][5-n_shots:]
           
            examples = [(eng_lines[idx].strip(), xmr_lines[idx].strip() if xmr_lines else "", tgt_lines[idx]) for idx in indices]
           
            prompt = build_prompt(examples, representation_type, n_shots, target_language)
           
            content = prompt + f"English: {eng_lines[i].strip()}\n"
            if representation_type and xmr_lines:
                content += f"{representation_type}: {xmr_lines[i].strip()}\n"
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
    directory = f'results/aya/{target_language.lower()}'
    os.makedirs(directory, exist_ok=True)
    
    eng_lines, umr_lines, amr_lines, tgt_lines = load_data(target_language)

    shot_configs = [0, 1, 3, 5]
    
    for n_shots in shot_configs:
        run(f'aya_{n_shots}_shot_umr.txt', n_shots, eng_lines, umr_lines, tgt_lines, 'Uniform Meaning Representation', target_language, directory)
        run(f'aya_{n_shots}_shot_amr.txt', n_shots, eng_lines, amr_lines, tgt_lines, 'Abstract Meaning Representation', target_language, directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, required=True)
    
    args = parser.parse_args()
    main(args.language)