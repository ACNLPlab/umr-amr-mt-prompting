# Incorporating Abstract Meaning Representation and Uniform Meaning Representation into In-Context Learning Strategies for Machine Translation

## Abstract
While in-context machine translation approaches lead to notable progress for low-resource languages, it is not yet clear which kinds of linguistic information in the prompts can consistently and reliably lead to performance gains. We evaluate whether semantic representations, specifically Abstract Meaning Representation (AMR) and its extension Uniform Meaning Representation (UMR)—can improve LLM-based MT via prompting. Across 12 languages, 7 models from 5 families, and multiple prompting strategies, we find that conclusions about their effectiveness are highly sensitive to the setting. We further observe that prompts using UMR tend to outperform those using AMR, particularly in few-shot scenarios. These results highlight the need for comprehensive evaluation for in-context learning work.

## Table of Contents
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Citation](#citation)

## Installation
```bash
git clone https://github.com/ACNLPlab/umr-amr-mt-prompting.git
cd nld
cd <instruction_type>

conda activate <model_type>
pip install -r requirements.txt
```

## Repository Structure

## Citation
If you find this work helpful, please cite:
```bibtext
not available yet
```