import re
import random
import json
from datasets import load_dataset
import sys
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import datetime

# --- Config ---
NUM_ORIGINALS = 10000  # number of original examples
SEED = 42
random.seed(SEED)

# Download NLTK resources if not already
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# SNLI label mapping
LABELS = {"entailment": "entailment", "neutral": "neutral", "contradiction": "contradiction"}

# Simple lexical swap dictionary (keep your existing)
LEXICAL_SWAP = {
    "cooking": "cleaning",
    "playing": "running",
    "happy": "sad",
    "sitting": "standing",
    "jumping": "falling",
}

if len(sys.argv) != 2:
    sys.exit("Usage: python generate_contrast_set.py [validation|test]")

dataset_split =  sys.argv[1] 

snli = load_dataset("snli")

if dataset_split == "validation":
    test_data = snli["validation"]
elif dataset_split == "test":
    test_data = snli["test"]
else:
    sys.exit("Invalid dataset split. Choose 'validation' or 'test'.")

# 2. Filter out all examples where the 'label' is -1
# The 'label' column is the gold label (entailment=0, neutral=1, contradiction=2)
# and the no-consensus examples are automatically assigned -1.
dataset_filtered = test_data.filter(lambda example: example['label'] != -1)

# Verify the result
print(f"Original number of rows: {len(test_data)}")
print(f"Filtered number of rows: {len(dataset_filtered)}")
print("Remaining unique labels:", set(dataset_filtered['label']))

NUM_ORIGINALS = len(dataset_filtered)
# Sample originals
originals = random.sample(list(dataset_filtered), NUM_ORIGINALS)

# --- Helper functions ---
def insert_negation(sentence):
    """Naive negation insertion for 'is' / 'are'."""
    if re.search(r"\b(is|are)\b", sentence):
        return re.sub(r'\b(is|are)\b', r'\1 not', sentence)
    return None

def lexical_swap_dict(sentence):
    """Replace a word using existing LEXICAL_SWAP dictionary."""
    for word, replacement in LEXICAL_SWAP.items():
        if re.search(r'\b' + re.escape(word) + r'\b', sentence):
            return re.sub(r'\b' + re.escape(word) + r'\b', replacement, sentence)
    return None

# WordNet helpers
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None

def get_synonyms_antonyms(word, pos=None):
    synonyms, antonyms = set(), set()
    for syn in wn.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
            if lemma.antonyms():
                for ant in lemma.antonyms():
                    antonyms.add(ant.name().replace("_", " "))
    synonyms.discard(word)
    antonyms.discard(word)
    return list(synonyms), list(antonyms)

def lexical_swap_wordnet(sentence, swap_prob=0.2, prefer_antonym=False):
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    new_tokens = []
    for word, tag in tagged:
        wn_pos = get_wordnet_pos(tag)
        if wn_pos and random.random() < swap_prob:
            synonyms, antonyms = get_synonyms_antonyms(word.lower(), wn_pos)
            swap_list = antonyms if prefer_antonym else synonyms
            if swap_list:
                new_word = random.choice(swap_list)
                if word[0].isupper():
                    new_word = new_word.capitalize()
                new_tokens.append(new_word)
                continue
        new_tokens.append(word)
    return " ".join(new_tokens) if new_tokens != tokens else None

def flip_label(original_label, edit_type):
    if edit_type == "negation":
        if original_label == 0: return 2
        elif original_label == 2: return 0
        else: return 1
    elif edit_type == "lexical_antonym":
        if original_label == 0: return 2
        elif original_label == 2: return 0
        else: return original_label
    else:  # synonym or dictionary swap
        return original_label

# --- Generate mix of original and contrast sets ---
original_examples = []
contrast_examples = []
num_edits = 0
num_neg_edits = 0
num_lexical_dict_swaps = 0
num_lexical_synonym_swaps = 0
num_lexical_antonym_swaps = 0

for ex in originals:
    premise = ex["premise"]
    hypothesis = ex["hypothesis"]
    label_orig = ex["label"]

    # Randomly choose edit type
    edit_type_choice = random.choices(
        ["negation", "lexical_dict", "lexical_synonym", "lexical_antonym", "none"],
        weights=[0.3, 0.0, 0.3, 0.3, 0.1],
        k=1
    )[0]

    if edit_type_choice == "negation":
        hyp_contrast = insert_negation(hypothesis)
        edit_type = "negation"
    elif edit_type_choice == "lexical_dict":
        hyp_contrast = lexical_swap_dict(hypothesis)
        edit_type = "lexical_dict"
    elif edit_type_choice == "lexical_synonym":
        hyp_contrast = lexical_swap_wordnet(hypothesis, prefer_antonym=False)
        edit_type = "lexical_synonym"
    elif edit_type_choice == "lexical_antonym":
        hyp_contrast = lexical_swap_wordnet(hypothesis, prefer_antonym=True)
        edit_type = "lexical_antonym"
    else:
        hyp_contrast = None
        edit_type = "none"

    if hyp_contrast is None:
        hyp_contrast = hypothesis
        label_contrast = label_orig
        edit_type = "none"
    else:
        label_contrast = flip_label(label_orig, edit_type)
        num_edits += 1
        if edit_type == "negation":
            num_neg_edits += 1
        elif edit_type == "lexical_dict":
            num_lexical_dict_swaps += 1
        elif edit_type == "lexical_antonym":
            num_lexical_antonym_swaps += 1
        elif edit_type == "lexical_synonym":
            num_lexical_synonym_swaps += 1

    # Save original and contrast
    original_examples.append({
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label_orig,
        "edit_type": edit_type
    })
    contrast_examples.append({
        "premise": premise,
        "hypothesis": hyp_contrast,
        "label": label_contrast,
        "edit_type": edit_type
    })

OUTPUT_DIR = "synthesized_data"
now = datetime.datetime.now()
timestamp = now.strftime("%Y%m%d%H%M%S")

original_filepath = f"{OUTPUT_DIR}/snli_original_{dataset_split}_{NUM_ORIGINALS}_{timestamp}.jsonl"
contrast_filepath = f"{OUTPUT_DIR}/snli_contrast_{dataset_split}_{NUM_ORIGINALS}_{timestamp}.jsonl"
# --- Save JSONL ---
with open(original_filepath, "w", encoding="utf-8") as f:
    for ex in original_examples:
        f.write(json.dumps(ex) + "\n")

with open(contrast_filepath, "w", encoding="utf-8") as f:
    for ex in contrast_examples:
        f.write(json.dumps(ex) + "\n")



print(f"Saved {len(original_examples)} original examples to {original_filepath}")
print(f"Saved {len(contrast_examples)} contrast examples to {contrast_filepath} with {num_edits} edits, {num_neg_edits} negation edits, {num_lexical_dict_swaps} lexical dict swaps, {num_lexical_synonym_swaps} lexical synonym swaps, {num_lexical_antonym_swaps} lexical antonym swaps.")
