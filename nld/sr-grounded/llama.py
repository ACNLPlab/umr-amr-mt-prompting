import re
import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json

checkpoint = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

##### OUTPUT CLEANING #####
def extract_translated_text(response: str, target_language: str) -> str:
    response = response.split(f"{target_language}:assistant\n\n", 1)[1].strip()
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
    with open('umr-amr-mt-prompting/lpp/all_eng_sent.txt', 'r', encoding='utf-8') as f:
        all_eng_lines = f.readlines()
    with open(f'{base_path}/{prefix[:3]}_sent.txt', 'r', encoding='utf-8') as f:
        tgt_lines = f.readlines()
    
    eng_lines = [line.strip() for line in eng_lines]
    all_eng_lines = [line.strip() for line in all_eng_lines]

    indices = []
    for line in eng_lines:
        if line in all_eng_lines:
            i = all_eng_lines.index(line)
            indices.append(i)
            all_eng_lines[i] = None
    
    umr_descriptions = []
    with open('../descriptions/umr_descriptions.txt', 'r', encoding='utf-8') as f:
        for line in f:
            sentences = json.loads(line.strip())
            umr_descriptions.append('\n'.join(sentences))
    
    amr_descriptions = []
    with open('../descriptions/amr_descriptions.txt', 'r', encoding='utf-8') as f:
        for line in f:
            sentences = json.loads(line.strip())
            amr_descriptions.append('\n'.join(sentences))
    
    paralllel_umr_descriptions = []
    paralllel_amr_descriptions = []
    for i in indices:
        paralllel_umr_descriptions.append(umr_descriptions[i])
        paralllel_amr_descriptions.append(amr_descriptions[i])
    
    return eng_lines, paralllel_umr_descriptions, paralllel_amr_descriptions, tgt_lines

def build_prompt(examples, representation_type, n_shots, target_language):    
    prompt = f"You will be provided with an English sentence and its {representation_type} described in natural language.\n"

    prompt += f"\nTask: Meaning-Preserving Translation"
    if representation_type:
        prompt += f" Using {representation_type}"
    prompt += "\n\n"
    
    prompt += "Instructions:\n"
    if representation_type:
        prompt += f"- Use the {representation_type} description as the full semantic structure.\n"
        prompt += "- Preserve predicate–argument relations.\n"
        prompt += "- Preserve negation, modifiers, quantities, and named entities.\n"
    prompt += f"- Please output ONLY the translation."

    if n_shots > 0:
        prompt += "\n\nExamples:"
    
    for idx, (eng, xmr_description, tgt) in enumerate(examples):
        prompt += "\n\n"
        prompt += f"English: {eng.strip()}\n"
        if representation_type and xmr_description:
            prompt += f"Input {representation_type} description:\n"
            prompt += f"{xmr_description.strip()}\n"
        prompt += f"{target_language}: {tgt.strip()}"

    prompt += f"\n\nTranslate the following English sentence into {target_language}:\n"

    return prompt

def translate(messages, target_language: str):
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(inputs, max_new_tokens=128)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)
    return extract_translated_text(response, target_language)

def run(output_file, n_shots, eng_lines, xmr_descriptions, tgt_lines, representation_type, target_language, directory):
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
           
            examples = [(eng_lines[idx].strip(), xmr_descriptions[idx].strip() if xmr_descriptions else "", tgt_lines[idx]) for idx in indices]
           
            prompt = build_prompt(examples, representation_type, n_shots, target_language)
           
            content = prompt + f"English: {eng_lines[i].strip()}\n"
            if representation_type and xmr_descriptions:
                content += f"Input {representation_type} description:\n"
                content += f"{xmr_descriptions[i].strip()}"
            content += f"\n{target_language}: "

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
    directory = f'results/llama/{target_language.lower()}'
    os.makedirs(directory, exist_ok=True)

    eng_lines, umr_descriptions, amr_descriptions, tgt_lines = load_data(target_language)

    shot_configs = [0, 1, 3, 5]
    
    for n_shots in shot_configs:
        run(f'llama_{n_shots}_shot_umr.txt', n_shots, eng_lines, umr_descriptions, tgt_lines, 'Uniform Meaning Representation', target_language, directory)
        run(f'llama_{n_shots}_shot_amr.txt', n_shots, eng_lines, amr_descriptions, tgt_lines, 'Abstract Meaning Representation', target_language, directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, required=True)
    
    args = parser.parse_args()
    main(args.language)