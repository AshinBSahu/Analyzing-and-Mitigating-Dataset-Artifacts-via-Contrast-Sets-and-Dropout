import os
import sys
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering
)
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns

if torch.backends.mps.is_available():
    # 1. Prioritize Apple Silicon MPS
    device = "mps"
elif torch.cuda.is_available():
    # 2. Fallback to NVIDIA CUDA
    device = "cuda"
else:
    # 3. Use CPU if neither GPU backend is found
    device = "cpu"

print(f"Using device: {device}")

def load_model_auto(model_path):
    """
    Automatically detects whether the model is a QA or NLI model
    and loads the correct architecture.
    """
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if "qa" in model_path.lower() or (
        config.architectures and any("QuestionAnswering" in arch for arch in config.architectures)
    ):
        model = AutoModelForQuestionAnswering.from_pretrained(model_path, config=config).to(device)
        task_type = "qa"
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config).to(device)
        task_type = "nli"

    model.eval()
    return model, tokenizer, task_type


def evaluate_cartography(model, tokenizer, dataset, task_type="nli", num_passes=5, batch_size=16):
    """
    Computes confidence, variability, and correctness for NLI task.
    Also records predicted labels for detailed analysis.
    """
    all_confidences, all_variabilities, all_correctness, all_preds = [], [], [], []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating for cartography"):
        batch = dataset.iloc[i: i + batch_size]

        if task_type == "nli":
            inputs = tokenizer(
                list(batch["premise"]),
                list(batch["hypothesis"]),
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)

            labels = torch.tensor(batch["label"].values).to(device)

            # Monte Carlo dropout-like evaluation
            model.train()
            probs_passes = []
            with torch.no_grad():
                for _ in range(num_passes):
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    probs_passes.append(probs.cpu().numpy())

            probs_passes = np.stack(probs_passes)
            mean_probs = probs_passes.mean(axis=0)
            var_probs = probs_passes.var(axis=0)
            preds = mean_probs.argmax(axis=-1)

            confidences = mean_probs.max(axis=-1)
            variabilities = var_probs.mean(axis=-1)
            correctness = (preds == labels.cpu().numpy()).astype(float)

            all_confidences.extend(confidences)
            all_variabilities.extend(variabilities)
            all_correctness.extend(correctness)
            all_preds.extend(preds)

    return np.array(all_confidences), np.array(all_variabilities), np.array(all_correctness), np.array(all_preds)

def plot_cartography(confidences, variabilities, correctness,
                     conf_threshold=0.8, var_threshold=0.01,
                     title="Cartography Map (ELECTRA on SNLI Contrast Set)"):

    plt.figure(figsize=(9, 7))
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Symbola', 'Arial']
    # Scatter plot
    sc = plt.scatter(
        variabilities, confidences,
        c=correctness, cmap="coolwarm",
        alpha=0.6, edgecolors="none"
    )

    # Axis labels
    plt.xlabel("Prediction Variability (Model Uncertainty)", fontsize=12)
    plt.ylabel("Mean Confidence (Predicted Probability)", fontsize=12)
    plt.title(title, fontsize=14, weight='bold')

    # Add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label("Correctness (0 = Wrong, 1 = Right)", fontsize=11)

    # Threshold lines for interpretive quadrants
    plt.axhline(conf_threshold, color="gray", linestyle="--", linewidth=1)
    plt.axvline(var_threshold, color="gray", linestyle="--", linewidth=1)

    # Region labels
    plt.text(var_threshold/2, conf_threshold + 0.05, "🟦 Easy / Stable\n(High conf, low var)",
             fontsize=10, ha="center", va="bottom", color="navy")
    plt.text(var_threshold + 0.015, conf_threshold + 0.05, "🟧 Overconfident\n(High conf, high var)",
             fontsize=10, ha="left", va="bottom", color="darkorange")
    plt.text(var_threshold/2, conf_threshold - 0.25, "🟥 Misunderstood\n(Low conf, low var)",
             fontsize=10, ha="center", va="bottom", color="darkred")
    plt.text(var_threshold + 0.015, conf_threshold - 0.25, "🟪 Hard / Ambiguous\n(Low conf, high var)",
             fontsize=10, ha="left", va="bottom", color="purple")

    # Optional grid for clarity
    plt.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.show()


def compute_edit_type_accuracy(df, dataset_name):
    """Compute accuracy per edit_type for a single dataset."""
    summary = (
        df.groupby("edit_type")["correctness"]
        .mean()
        .reset_index()
        .rename(columns={"correctness": "accuracy"})
    )
    summary["dataset"] = dataset_name
    return summary


def plot_accuracy_original_contrast_comparison(original_summary, contrast_summary, title=None, filename=None, save_dir='plot_outputs',data_dir='data_outputs'):
    """
    Plot side-by-side bar chart comparing accuracies by edit type.
    Displays the exact accuracy value (data label) above each bar for precise reading.
    This fulfills the requirement to mark the accuracy number for each bar.
    """
    
    # --- 1. Data Preparation ---
    # Concatenate the summaries
    combined = pd.concat([original_summary, contrast_summary], ignore_index=True)
    
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    if not filename:
        filename = "accuracy_comparison"
    save_filename = f"{filename.replace(' ', '_').lower()}.png"


    # --- 2. Plotting ---
    plt.figure(figsize=(9, 6))
    
    # Create the bar plot using the professional 'colorblind' palette
    ax = sns.barplot(
        data=combined,
        x="edit_type",
        y="accuracy",
        hue="dataset",
        palette="colorblind",
        edgecolor="black",
        saturation=0.8
    )
    
    # --- 3. Add Data Labels (Annotations) for Precision ---
    # This explicitly marks the accuracy number for each bar.
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            
            # Skip bars with no height (or tiny height)
            if height < 0.001:
                continue

            ax.annotate(
                f'{height:.3f}',  # Format the accuracy to 3 decimal places (ACL standard)
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),    # 3 points vertical offset
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=9,
                color='black',
                fontweight='bold'
            )


    # --- 4. Enhance Aesthetics ---
    ax.set_title(title if title else "Model Accuracy by Edit Type (Original vs. Contrast)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Edit Type", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0.0, 1.0)
    #ax.legend(title="Dataset", loc="upper right", frameon=True, fancybox=True) 
    ax.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fancybox=True)
    
    # Use a finer grid to help read values to meet the requirement for "dotted lines"
    ax.grid(axis='y', linestyle=':', alpha=0.5, which='major')
    
    plt.tight_layout()
    

    # --- 5. Save and Show ---
    full_save_path = os.path.join(save_dir, save_filename)
    plt.savefig(full_save_path, dpi=300)
    print(f"Plot successfully saved to: {full_save_path}")

    #plt.show()

    # --- 6. Print Summary Table ---
    print("\n=== Accuracy by Edit Type (Original vs Contrast) ===")
    summary_table = combined.pivot(index="edit_type", columns="dataset", values="accuracy")
    print(summary_table)

    # --- 7. Save Summary Table to File ---
    
    # Define the content and path for the summary file
    summary_header = "=== Accuracy by Edit Type (Original vs Contrast) ===\n"
    summary_content = summary_table.to_string()
    
    summary_filename = f"{filename.replace(' ', '_').lower()}_summary.txt"
    summary_filepath = os.path.join(data_dir, summary_filename)
    
    # Write the header and the pandas table string to the file
    try:
        with open(summary_filepath, "w") as f:
            f.write(summary_header)
            f.write(summary_content)
            f.write("\n")
        print(f"Summary table successfully flushed to: {summary_filepath}")
    except IOError as e:
        print(f"Error writing summary file: {e}")

    return combined

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_cartography_original_vs_contrast(original_df, contrast_df,
                                       title="original vs Contrast Cartography",
                                       filename=None, save_dir='plot_outputs'):
    """
    Plots side-by-side faceted cartography maps for original and contrast datasets.
    Each tile corresponds to an edit_type, dots colored by correctness.
    Left: original, Right: Contrast.
    """

    # Copy and label datasets
    original_df = original_df.copy()
    contrast_df = contrast_df.copy()
    original_df["dataset_type"] = "original"
    contrast_df["dataset_type"] = "Contrast"

    # Concatenate datasets
    df = pd.concat([original_df, contrast_df], ignore_index=True)

    # Map correctness to label for coloring
    df["correctness_label"] = df["correctness"].map({0: "Wrong", 1: "Correct"})

    # Determine unique edit types and dataset types
    edit_types = df["edit_type"].unique()
    dataset_types = df["dataset_type"].unique()

    # Create FacetGrid: columns = dataset_type, rows = edit_type
    g = sns.FacetGrid(df, row="edit_type", col="dataset_type", sharex=True, sharey=True,
                      height=3.5, aspect=1.2, hue="correctness_label",
                      palette="colorblind")

    # Plot scatter
    g.map_dataframe(sns.scatterplot, x="variability", y="confidence", alpha=0.7, s=40, edgecolor="none")

    # Axis labels and title
    g.set_axis_labels("Prediction Variability", "Mean Confidence")
    g.fig.subplots_adjust(top=0.90)
    g.fig.suptitle(title, fontsize=16)

    # Annotate total per tile
    for ax, (row_val, col_val) in zip(g.axes.flat, [(r, c) for r in edit_types for c in dataset_types]):
        count = len(df[(df["edit_type"] == row_val) & (df["dataset_type"] == col_val)])
        ax.text(
            0.95, 0.02,
            f"N={count:,}",
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=9,
            color="gray",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.3")
        )



    # Add legend for correctness
    #g.add_legend(title="Correctness", loc = "upper right")
    # Adjusted legend location for better fit outside the plots
    g.add_legend(title="Prediction Outcome", loc="upper right")


    plt.tight_layout()

    # Save figure
    if filename:
        save_path = os.path.join(save_dir, f"{filename.replace(' ', '_').lower()}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Plot successfully saved to: {save_path}")

    #plt.show()
    plt.close()

if __name__ == "__main__":

    if len(sys.argv) != 4:
        sys.exit("Usage: python generate_contrast_set.py <model> <original data> <edited data>")

    model_path = sys.argv[1]
    original_data_path = sys.argv[2]
    contrast_data_path = sys.argv[3]

    if "validation" in original_data_path:
        dataset_split_type = "validation-dataset"
    elif "test" in original_data_path:
        dataset_split_type = "test-dataset"
    else:
        dataset_split_type = "unknown-dataset"

    print (f"Dataset Split Type: {dataset_split_type}")

    
    print (f"Model Path: {model_path}")
    print (f"original Data Path: {original_data_path}")
    print (f"Contrast Data Path: {contrast_data_path}")

    # Define the target directory and filename
    save_dir = 'plot_outputs'
    # os.path.splitext returns a tuple: (root, ext)
    root_original, extension_original = os.path.splitext(original_data_path)
    root_contrast, extension_contrast = os.path.splitext(contrast_data_path)

    root_original = os.path.basename(root_original)
    root_contrast = os.path.basename(root_contrast)

    print (f"Root original: {root_original}, Root Contrast: {root_contrast}")
    
    # os.path.basename() extracts the last part of the path
    normalized_path = os.path.normpath(model_path)
    model_name = os.path.basename(normalized_path)
    
    print(f"The extracted model name is: {model_name}")
    # Output: The extracted model name is: checkpoint-12500

    
    # Create the directory if it doesn't exist
    # This is a critical step to prevent a FileNotFoundError
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    
    #model_path = "./trained_model/checkpoint-12500/"
    #original_data_path = "./snli_original_validation_10000.jsonl"       # Original (no edits)
    #contrast_data_path = "./snli_contrast_validation_10000.jsonl" # Edited dataset

    model, tokenizer, task_type = load_model_auto(model_path)
    print(f"Loaded {task_type.upper()} model from {model_path}")

    print("\n=== Evaluating original Dataset ===")
    original_df = pd.read_json(original_data_path, lines=True)
    p_conf, p_var, p_corr, p_preds = evaluate_cartography(model, tokenizer, original_df, task_type)
    original_df["confidence"] = p_conf
    original_df["variability"] = p_var
    original_df["correctness"] = p_corr
    original_df["predicted_label"] = p_preds

    print("\n=== Evaluating Contrast (Edited) Dataset ===")
    contrast_df = pd.read_json(contrast_data_path, lines=True)
    c_conf, c_var, c_corr, c_preds = evaluate_cartography(model, tokenizer, contrast_df, task_type)
    contrast_df["confidence"] = c_conf
    contrast_df["variability"] = c_var
    contrast_df["correctness"] = c_corr
    contrast_df["predicted_label"] = c_preds

    # Cartography plot for contrast dataset
    #plot_cartography(c_conf, c_var, c_corr)

    # Compute per-edit-type accuracy
    original_summary = compute_edit_type_accuracy(original_df, "original")
    contrast_summary = compute_edit_type_accuracy(contrast_df, "Contrast (Edited)")

    # Plot comparison
    print ("Plot Accuracy for original vs contrast")
    plot_accuracy_original_contrast_comparison(original_summary, contrast_summary,title="original vs Contrast Accuracy by Edit Type",filename=f"{model_name}_{dataset_split_type}_original vs Contrast Accuracy by Edit Type", save_dir=save_dir)


    print ("Plot Cartography for original vs contrast cartography")
    plot_cartography_original_vs_contrast(original_df, contrast_df,title ="original vs contrast cartography", filename=f"{model_name}_{dataset_split_type}_original vs contrast cartography")

    # Save all results
    original_df.to_csv(f"data_outputs/{model_name}_{dataset_split_type}_original_cartography_results.csv", index=False)
    contrast_df.to_csv(f"data_outputs/{model_name}_{dataset_split_type}_contrast_cartography_results.csv", index=False)
    print("\n✅ Saved per-example results for original and contrast datasets.")
