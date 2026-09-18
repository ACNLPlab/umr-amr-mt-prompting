import os
import sys
from sentence_transformers import SentenceTransformer, util

THRESHOLD = 0.5

if len(sys.argv) < 2:
    sys.exit("Usage: python validate.py <language>  (e.g. Chinese, or its first 4 letters: chin)")

target_lang = sys.argv[1]

prefix = target_lang.lower()[:4]
folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{prefix}_experiment_data')

# Load multilingual model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

with open(os.path.join(folder, 'eng_sent.txt'), 'r', encoding='utf-8') as f:
    eng = f.readlines()
with open(os.path.join(folder, f'{prefix[:3]}_sent.txt'), 'r', encoding='utf-8') as f:
    tgt = f.readlines()
assert len(eng) == len(tgt), "eng_sent.txt and target sentence file are not parallel"

misaligned = []
for idx in range(len(eng)):
    # Encode both sentences
    emb_en = model.encode(eng[idx])
    emb_tgt = model.encode(tgt[idx])

    # Compute cosine similarity
    similarity = util.pytorch_cos_sim(emb_en, emb_tgt).item()

    status = "✓" if similarity >= THRESHOLD else "⚠️"
    print(f"{status} Line {idx+1}: {similarity:.3f}")
    if similarity < THRESHOLD:
        misaligned.append((idx+1, similarity))
    else:
        print(f"   EN: {eng[idx][:60]}...")
        print(f"   TG: {tgt[idx][:60]}...")

print(f"\n\nPotential misalignments (similarity < {THRESHOLD}): {len(misaligned)}")
for line_num, sim in misaligned:
    print(f"\n  Line {line_num}: {sim:.3f}")
    print(f"    EN: {eng[line_num-1].strip()}")
    print(f"    TG: {tgt[line_num-1].strip()}")
