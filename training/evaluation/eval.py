import argparse
from generate_and_eval import calculate_scores

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained seq2seq model.")
    parser.add_argument("--hypothesis_file", type=str, required=True, help="Path to the hypothesis text file (one sentence per line).")
    parser.add_argument("--reference_file", type=str, required=True, help="Path to the reference/target text file (one sentence per line).")
    
    args = parser.parse_args()

    # --- 1. Load Data ---
    with open(args.hypothesis_file, "r", encoding="utf-8") as f:
        hypothesis_sents = [line.strip() for line in f.readlines()]
    with open(args.reference_file, "r", encoding="utf-8") as f:
        reference_sents = [line.strip() for line in f.readlines()]

    # --- 2. Calculate Scores ---
    scores = calculate_scores(hypothesis_sents, reference_sents)

    # --- 3. Print Results ---
    print("\n" + "="*25)
    print("      EVALUATION RESULTS")
    print("="*25)
    print(f"  BLEU Score    : {scores['sacrebleu']['score']:.2f}")
    print(f"  chrF++ Score  : {scores['chrf++']['score']:.2f}")
    print(f"  CER Score     : {scores['cer']:.4f} (Lower is better)")
    print("="*25)

if __name__ == "__main__":
    main()