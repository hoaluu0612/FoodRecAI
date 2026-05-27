import argparse
from pathlib import Path
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
from data_processor import data_splitting
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "result"

# Ensure UTF-8 encoding for console output to handle Vietnamese characters properly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def write_report_txt(output_path, lines):
    # Write the evaluation report to a text file with UTF-8 encoding
    with open(output_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        f.write("\n" + "="*60 + "\n\n")

def evaluate_svm_model(data_path, model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Cannot find model file: {model_path}")

    # Load the test data (test_size=0.2 and random_state=42 are handled inside data_processor)
    X_train_raw, X_test_raw, y_train, y_test = data_splitting(data_path=data_path)
    
    # Load the trained pipeline from disk
    pipeline = joblib.load(model_path)
    y_pred = pipeline.predict(X_test_raw)

    # Calculate the evaluation metrics for the report
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "weighted_precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
    }
    
    # Calculate evaluation stats for reporting
    eval_stats = {
        "total_used": len(X_train_raw) + len(X_test_raw),
        "train_size": len(X_train_raw),
        "test_size": len(X_test_raw)
    }
    
    # Define the class labels for the classification report and confusion matrix
    labels = ["negative", "neutral", "positive"]
    
    # Generate the classification report and confusion matrix
    report = classification_report(y_test, y_pred, labels=labels, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    
    return metrics, report, matrix, eval_stats, labels


def draw_simple_confusion_matrix(matrix, labels, output_path, model_name="SVM Model"):
    # Create the confusion matrix plot
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    plt.rcParams['font.family'] = 'serif'   
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    
    # Draw the pink background for the confusion matrix area with some transparency
    ax.matshow(matrix, cmap="pink", alpha=0.4)
    
    # Set the ticks to be at the center of each cell
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    
    # Set the tick labels with the specified font and size
    ax.set_xticklabels(labels, fontsize=9, fontname='Times New Roman')
    ax.set_yticklabels(labels, fontsize=9, fontname='Times New Roman')
    
    # Move the x-axis ticks to the bottom of the plot
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(len(labels)) + 0.5, minor=True)
    ax.set_yticks(np.arange(len(labels)) + 0.5, minor=True)
    
    # Draw the grid lines for the minor ticks to create the cell borders
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1.5)
    
    # Turn off the minor ticks themselves so only the grid lines are visible
    ax.tick_params(which="minor", bottom=False, left=False)

    # Calculate the total sum of the confusion matrix for percentage calculations
    sum_matrix = np.sum(matrix)
    for i in range(matrix.shape[0]):       
        for j in range(matrix.shape[1]):   
            value = matrix[i, j]
            percentage = (value / sum_matrix) * 100
            cell_text = f"{value}\n({percentage:.1f}%)"
            ax.text(j, i, cell_text, ha="center", va="center", 
                    fontsize=10, fontweight="bold", color="#111111")

    # Set the title and axis labels with the specified font and size
    ax.set_title(f'CONFUSION MATRIX\n({model_name})', fontsize=11, fontweight='bold', pad=15, fontname='Times New Roman')
    ax.set_xlabel('Predicted Label', fontsize=10, labelpad=10, fontname='Times New Roman')
    ax.set_ylabel('True Label', fontsize=10, labelpad=10, fontname='Times New Roman')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"-> Drawed! Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="This module evaluates the SVM model and exports a .txt report file.")
    parser.add_argument("--data", default="reviews_clean.jsonl")
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()

    # Determine the actual model path, checking both the outputs directory and the current directory
    actual_model_path = Path(__file__).resolve().parent / "outputs" / args.model_name
    if not actual_model_path.exists():
        actual_model_path = Path(__file__).resolve().parent / args.model_name

    # Evaluate the model and get all the necessary information for the report
    metrics, report, matrix, eval_stats, labels = evaluate_svm_model(
        data_path=args.data,
        model_path=actual_model_path
    )

    # Prepare the lines for the evaluation report text file
    report_lines = [
        "===== DATA =====",
        f"Total used: {eval_stats['total_used']}",
        f"Train size: {eval_stats['train_size']}",
        f"Test size: {eval_stats['test_size']}",
        f"Tested Model File: {actual_model_path.name}",
        "",
        "===== METRICS =====",
        *[f"{name}: {value:.4f}" for name, value in metrics.items()],
        "",
        "===== CLASSIFICATION REPORT =====",
        report,
        "===== CONFUSION MATRIX =====",
        str(matrix),
    ]
    
    # Create the output directory for evaluation results if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    report_path = OUTPUT_DIR / "svm_evaluation_report.txt"
    
    # Append the evaluation report to the text file
    write_report_txt(report_path, report_lines)

    # Draw the confusion matrix and save it as an image file
    display_labels = ["Negative", "Neutral", "Positive"]
    image_out_path = OUTPUT_DIR / f"matrix_{Path(args.model_name).stem}.png"
    draw_simple_confusion_matrix(matrix, display_labels, image_out_path, model_name=Path(args.model_name).stem)

    # Print the evaluation report to the console
    print("\n".join(report_lines))
    print(f"\n Successfully exported evaluation report to {report_path}")


if __name__ == "__main__":
    main()