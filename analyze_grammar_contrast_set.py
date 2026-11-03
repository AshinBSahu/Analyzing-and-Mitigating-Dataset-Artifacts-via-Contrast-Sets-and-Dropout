"""
Analyze grammatical correctness of hypotheses per edit type
using the CoLA acceptability classifier.
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ===== CONFIGURATION =====
INPUT_FILE = "data_outputs/model-baseline_validation-dataset_contrast_cartography_results.csv"
PREMISES_COLUMN = "premise"
TEXT_COLUMN = "hypothesis"
EDIT_COLUMN = "edit_type"
THRESHOLD = 0.3  # CoLA probability > 0.3 ⇒ grammatically correct
BATCH_SIZE = 32

# ===== LOAD DATA =====
print("Loading CSV...")
df = pd.read_csv(INPUT_FILE)
df = df[[PREMISES_COLUMN,TEXT_COLUMN, EDIT_COLUMN]].dropna()

# ===== LOAD MODEL =====
print("Loading CoLA model...")
model_name = "textattack/roberta-base-CoLA"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# ===== FUNCTION TO COMPUTE CoLA SCORES =====
def cola_acceptability(sentences, batch_size=BATCH_SIZE):
    probs = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            logits = model(**inputs).logits
        prob = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # P(acceptable)
        probs.extend(prob)
    return probs

# ===== RUN ANALYSIS =====
print("Evaluating grammatical acceptability...")
df["cola_prob"] = cola_acceptability(df[TEXT_COLUMN].astype(str).tolist())
df["grammatically_correct"] = df["cola_prob"] > THRESHOLD

# ===== AGGREGATE STATS =====
stats = (
    df.groupby(EDIT_COLUMN)["grammatically_correct"]
      .agg(["count", "sum"])
      .rename(columns={"count": "total", "sum": "correct"})
)
stats["incorrect"] = stats["total"] - stats["correct"]
stats["%_correct"] = (100 * stats["correct"] / stats["total"]).round(1)
stats["%_incorrect"] = (100 * stats["incorrect"] / stats["total"]).round(1)

print("\n===== GRAMMATICALITY STATS BY EDIT TYPE =====")
print(stats)
print("=============================================\n")

# --- 7. Save Summary Table to File ---

# Define the content and path for the summary file
summary_header = "\n===== GRAMMATICALITY STATS BY EDIT TYPE =====\n"
summary_content = stats.to_string()

summary_filename = "data_outputs/contrast_grammar_results_summary.txt"

# Write the header and the pandas table string to the file
try:
    with open(summary_filename, "w") as f:
        f.write(summary_header)
        f.write(summary_content)
        f.write("\n")
    print(f"Summary table successfully flushed to: {summary_filename}")
except IOError as e:
    print(f"Error writing summary file: {e}")
    
# ===== SAMPLE EXAMPLES =====
examples = {}
for edit in df[EDIT_COLUMN].unique():
    sub = df[df[EDIT_COLUMN] == edit]
    correct_ex = sub[sub["grammatically_correct"]].sample(
        min(2, len(sub[sub["grammatically_correct"]])), random_state=42
    )[TEXT_COLUMN].tolist()
    incorrect_ex = sub[~sub["grammatically_correct"]].sample(
        min(2, len(sub[~sub["grammatically_correct"]])), random_state=42
    )[TEXT_COLUMN].tolist()
    examples[edit] = {"correct": correct_ex, "incorrect": incorrect_ex}

# ===== PRINT EXAMPLES =====
for edit_type, ex in examples.items():
    print(f"\n### {edit_type.upper()} ###")
    print("Correct examples:")
    for s in ex["correct"]:
        print(f"   - {s}")
    print("Incorrect examples:")
    for s in ex["incorrect"]:
        print(f"   - {s}")

# ===== SAVE OPTIONAL OUTPUT =====
df.to_csv("data_outputs/contrast_grammar_results.csv", index=False)
print("\nSaved detailed results → data_outputs/contrast_grammar_results.csv")
