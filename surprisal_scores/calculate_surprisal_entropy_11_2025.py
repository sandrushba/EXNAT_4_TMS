"""
Calculate surprisal and entropy for EXNAT_4_TMS

Input: Python file with texts (same as for Psychopy experiment)

Output: Text file with surprisal and entropy values on a given context window

Written by Sandra Martin
Last modifiend: 2025/11/05
"""
import os
import re
import ast
import string
import warnings
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyprojroot.here import here

# Load German GPT-2
tok = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
model = AutoModelForCausalLM.from_pretrained("dbmdz/german-gpt2")
model.eval()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)

def enforce_two_words(text: str) -> str:
    # Strip and split on whitespace; keep exactly the last two orthographic words
    parts = text.strip().split()
    if len(parts) < 2:
        raise ValueError("Need at least two words in context.")
    last_two = parts[-2:]
    # Re-join with single spaces; GPT-2 expects correct mid-sentence spaces
    return " ".join(last_two)

@torch.no_grad()
def surprisal_entropy_two_word_context(context_words: str, target_word: str, model, tok, device):
    # Guard: warn if not exactly two words, but continue
    parts = context_words.strip().split()
    if len(parts) != 2:
        warnings.warn(f"context_words contains {len(parts)} words instead of 2: '{context_words}'")

    context = " ".join(parts)
    ctx_ids = tok.encode(context, add_special_tokens=False)
    ctx = torch.tensor([ctx_ids], device=device)

    # Entropy before any part of the target is revealed
    logits_ctx = model(input_ids=ctx).logits[:, -1, :]
    probs_ctx = torch.softmax(logits_ctx, dim=-1)
    log_probs_ctx = torch.log_softmax(logits_ctx, dim=-1)
    entropy_token = float((-probs_ctx * log_probs_ctx).sum().item())

    # Encode target with correct leading space
    target_text = " " + target_word
    tgt_ids = tok.encode(target_text, add_special_tokens=False)

    # Accumulate word-level log-prob across subtokens
    total_logprob = 0.0
    running_ids = ctx_ids[:]
    for tid in tgt_ids:
        x = torch.tensor([running_ids], device=device)
        logits = model(input_ids=x).logits[:, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        lp = float(log_probs[0, tid].item())
        total_logprob += lp
        running_ids.append(tid)

    surprisal = -total_logprob
    return surprisal, entropy_token

# Example usage:
#full_text_context = "Virginia Stephen besuchte keine Schule, sondern erhielt von Hauslehrern und ihrem Vater Privatunterricht. Sie war beeindruckt von der schriftstellerischen Arbeit ihres"
#two_word_context = enforce_two_words(full_text_context)  # ensures exactly 2 words
#result = surprisal_entropy_two_word_context(two_word_context, "Vaters")
#print(result)

# Read in data
texts = {}
#base_dir = "/data/tu_martin_cloud/EXNAT/EXNAT_4_TMS/Results/EXNAT_4_TMS_Analysis/word_frequencies/"
base_dir = os.path.dirname(__file__)
rel_path = os.path.join(base_dir, 'EXNAT_4_TMS_texts_MC_Qs.py')

with open(rel_path, 'r', encoding='utf-8') as f:
    file_content = f.read()

# Match any variable pattern like 'text_01 = [...]'
matches = re.findall(r'(text_\d+)\s*=\s*(\[[^\]]+\])', file_content)

for name, list_str in matches:
    word_list = ast.literal_eval(list_str)  # Safely convert string list to actual Python list
    texts[name] = word_list

# Flatten dict into a DataFrame
rows = []
for text_nr, words in texts.items():
    for idx, word in enumerate(words):
        # Remove punctuation
        clean_word = word.translate(str.maketrans('', '', string.punctuation))
        rows.append({
            'text_nr': text_nr,
            'trial_nr': idx+1,
            'word': word,
            'word_no_punct': clean_word
        })
texts_df = pd.DataFrame(rows)

# add 2 columns to texts_df for surprisal scores and entropies
texts_df['surprisal_2_words_context'] = None
texts_df['entropy_2_words_context'] = None

n_context_words = 2
previous_text_nr = texts_df["text_nr"][0]

for word_idx, word in enumerate(texts_df["word_no_punct"]):
    if texts_df["text_nr"][word_idx] != previous_text_nr:
        previous_text_nr = texts_df["text_nr"][word_idx]
        if texts_df["trial_nr"][word_idx] <= n_context_words:
            continue
    if texts_df["trial_nr"][word_idx] > n_context_words:
        print(word_idx, texts_df["trial_nr"][word_idx], word)
        previous_context = ' '.join(texts_df["word_no_punct"][word_idx - n_context_words:word_idx])
        curr_surprisal, curr_entropy = surprisal_entropy_two_word_context(
            previous_context,  # string of two words
            word,              # actual next word
            model,
            tok,
            device
        )
        texts_df.at[word_idx, 'surprisal_2_words_context'] = curr_surprisal
        texts_df.at[word_idx, 'entropy_2_words_context'] = curr_entropy

# save the df:
texts_df.to_csv(here("surprisal_scores/vanilla_2words_surprisal_entropy_48Texts_11_2025.csv"), encoding="utf-8-sig")