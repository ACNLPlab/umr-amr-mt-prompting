import penman
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from textwrap import dedent
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key="OPENAI_API_KEY")
model_id = "gpt-4o-mini"

with open('relation_mapping.json', 'r') as f:
    relation_mapping = json.load(f)

with open('entity_mapping.json', 'r') as f:
    entity_mapping = json.load(f)

def xmr_to_triplets(xmr):
    graph = penman.decode(xmr)
    instances = {source: target for source, role, target in graph.triples if role == ":instance"}
    output = []
    for i, (source, role, target) in enumerate(graph.triples):
        if role != ":instance":
            source = instances[source] if source in instances else source
            target = instances[target] if target in instances else target
            output.append((source, role, target))
    return output

def triplets_to_sentences(triplets, relation_mapping, entity_mapping):
    sentences = []
    
    for source, role, target in triplets:
        entity1 = entity_mapping[source] if source in entity_mapping else source
        entity2 = entity_mapping[target] if target in entity_mapping else target
        
        role = role.lower()
        if entity1 in ("amr-unknown", "umr-unknown") or entity2 in ("amr-unknown", "umr-unknown"):
            sentence = f"{target} indicates a question about the {role.replace(':', '')} of {entity1}."
            sentences.append(sentence)
        else:
            sentences.append(relation_mapping[role].format(entity1=entity1, entity2=entity2))
        
    return sentences
    
def create_prompt(original_sentence, sub_sentences):
    system_prompt = dedent("""\
        Your task is to polish a collection of sub sentences and rewrite them based on the original sentence to make them read more like natural language.
        Please output your answer according to the examples.
        ###
        Example 1:
        input_original_sentence: ""He presented his research at the meeting yesterday."
        input_sub_sentences: "
        0 person is the doer of present-01(give).
        1 3rd is the grammatical person of person.
        2 singular is the grammatical number of person.
        3 thing is the object of present-01(give).
        4 thing is the object of research-01(study very hard).
        5 person is the doer of research-01(study very hard).
        6 present-01(give) takes place at meet-01(arrive at, achieve).
        7 meet-01(arrive at, achieve) has aspect process(unspecified type of process).
        8 present-01(give) is under the temporal circumstances of yesterday.
        9 present-01(give) has aspect performance(process that ends and reaches a result state).
        10 present-01(give)'s modal strength is full-affirmative(full affirmative support; complete certainty that the event occurs).        
        output_sentences: "
        0 A person is the one who presented
        1 The person is referred to in the third person.
        2 The person is singular.
        3 Something is being presented.
        4 This thing is research.
        5 This research is done by the person.
        6 The presentation took place at a meeting.
        7 The meeting is described as a process, without specifying its exact type.
        8 This presentation took place yesterday.
        9 The presentation is completed and reaches a result.
        10 There is full certainty that this presentation took place."
        ###
        Example 2:
        input_original_sentence: "Are you still there?"
        input_sub_sentences: "
        0 you is the object of be-located-at-91(reification of :location).
        1 there is the indirect object of be-located-at-91(reification of :location).
        2 be-located-at-91(reification of :location) is described as still.
        3 amr-unknown indicates a question about the polarity of be-located-at-91(reification of :location)."
        output_sentences: "
        0 "you" is the one whose location is being described. 
        1 "there" is the location.
        2 The state of being located is described as continuing ("still")
        3 This is a yes/no question about whether that state holds.\"""")
    
    if isinstance(sub_sentences, list):
        sub_sentences_str = "\n".join([f"{i} {s}" for i, s in enumerate(sub_sentences)])
    else:
        sub_sentences_str = sub_sentences
    
    user_prompt = f"input_original_sentence: {original_sentence.strip()}"
    user_prompt += f"\ninput_sub_sentences:"
    user_prompt += f"\n{sub_sentences_str}"
    user_prompt += "\noutput_sentences:"
    
    return system_prompt, user_prompt

def polish(original_sentence, sub_sentences):
    system_prompt, user_prompt = create_prompt(original_sentence, sub_sentences)

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    )
    
    response_text = response.choices[0].message.content
    return response_text

def xmr_to_nld(sentence, xmr):
    triplets = xmr_to_triplets(xmr)
    sub_sentences = triplets_to_sentences(triplets, relation_mapping, entity_mapping)
    output = polish(sentence, sub_sentences)
    return output.strip()

# Load data
with open('umr-amr-mt-prompting/lpp/all_eng_sent.txt', 'r', encoding='utf-8') as f:
    lpp_eng_lines = f.readlines()
with open('umr-amr-mt-prompting/lpp/all_eng_umrs.txt', 'r', encoding='utf-8') as f:
    lpp_umr_lines = f.readlines()
with open('umr-amr-mt-prompting/lpp/all_eng_amrs.txt', 'r', encoding='utf-8') as f:
    lpp_amr_lines = f.readlines()

if __name__ == "__main__":
    import os
    import json
    import traceback
    
    output_dir = "descriptions"
    os.makedirs(output_dir, exist_ok=True)
    
    umr_output_file = os.path.join(output_dir, "umr_descriptions.txt")
    amr_output_file = os.path.join(output_dir, "amr_descriptions.txt")
    
    # Initialize files
    open(umr_output_file, 'w', encoding='utf-8').close()
    open(amr_output_file, 'w', encoding='utf-8').close()
    
    for idx, (eng_sent, umr_graph, amr_graph) in enumerate(tqdm(zip(lpp_eng_lines, lpp_umr_lines, lpp_amr_lines), total=len(lpp_eng_lines), desc="Processing")):
        if not umr_graph.strip():
            with open(umr_output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps([]) + "\n")
            with open(amr_output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps([]) + "\n")
            continue
        
        # Process UMR
        umr_result = xmr_to_nld(eng_sent, umr_graph)
        umr_sentences = [s.strip() for s in umr_result.split('\n') if s.strip()]
        with open(umr_output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(umr_sentences) + "\n")
        
        # Process AMR
        amr_result = xmr_to_nld(eng_sent, amr_graph)
        amr_sentences = [s.strip() for s in amr_result.split('\n') if s.strip()]
        with open(amr_output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(amr_sentences) + "\n")