import argparse
from pathlib import Path
import joblib
from data_processor import data_splitting
from model import build_svm_pipeline

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

def train_model(data_path, test_size, random_state, c_param, max_features, version_name):
    # Data splitting 
    X_train_raw, X_test_raw, y_train, y_test = data_splitting(
        data_path=data_path, test_size=test_size, random_state=random_state
    )
    
    # Build SVM pipeline
    pipeline = build_svm_pipeline(max_features=max_features, c_param=c_param, random_state=random_state)
    
    print(f"-> Training Processing ... (C={c_param}, Max_Features={max_features}) for the version: {version_name}...")
    
    # Train model
    pipeline.fit(X_train_raw, y_train)
    print("-> Train Successfully")
    
    # Save the trained pipeline to disk
    OUTPUT_DIR.mkdir(exist_ok=True)
    model_path = OUTPUT_DIR / f'svm_pipeline_{version_name}.pkl'
    joblib.dump(pipeline, model_path)
    print(f"-> Pipeline saved to {model_path}")
    
    # Calculate stats for report
    stats = {
        "data_path": str(Path(data_path).resolve()),
        "train_size": len(X_train_raw),
        "test_size": len(X_test_raw),
        "total_used": len(X_train_raw) + len(X_test_raw),
        "model_path": str(model_path.resolve()),
        "c_param": c_param,
        "max_features": max_features,
        "version": version_name
    }
    
    return pipeline, stats


def main():
    # Set up argument parser for command-line execution
    parser = argparse.ArgumentParser(description="Mô-đun huấn luyện SVM nâng cao.")
    parser.add_argument("--data", default="reviews_clean.jsonl")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--version", default="C1.0")
    args = parser.parse_args()

    # Train the model and get stats for reporting
    _, stats = train_model(
        data_path=args.data, 
        test_size=args.test_size, 
        random_state=args.random_state, 
        c_param=args.c, 
        max_features=args.max_features, 
        version_name=args.version
    )

    # Print the training report
    print("\n" + "="*20 + " TRAIN REPORT " + "="*20)
    print(f"Data file:     {stats['data_path']}")
    print(f"Total used:    {stats['total_used']} rows")
    print(f"Train size:    {stats['train_size']} rows")
    print(f"Test size:     {stats['test_size']} rows")
    print(f"SVM Param C:   {stats['c_param']}")
    print(f"TFIDF Max:     {stats['max_features']} features")
    print(f"Saved Version: {stats['version']}")
    print(f"Model Path:    {stats['model_path']}")
    print("="*54 + "\n")

if __name__ == "__main__":
    main()