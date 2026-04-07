import os
import random

random.seed(1234)

N = 100

def split_csv(input_csv_path, new_path_prefix):
    # Split off N words for out-of-vocab evaluation
    output_dir = os.path.dirname(input_csv_path)
    eval_csv_path = os.path.join(output_dir, f"{new_path_prefix}_eval_cleaned.csv")
    remaining_csv_path = os.path.join(output_dir, f"{new_path_prefix}_train_cleaned.csv")

    with open(input_csv_path, "r", encoding="utf-8") as infile, \
            open(eval_csv_path, "w", encoding="utf-8") as eval_file, \
            open(remaining_csv_path, "w", encoding="utf-8") as train_file:
        lines = infile.readlines()
        header = lines[0]
        eval_file.write(header)

        # Shuffle lines to ensure randomness
        data_lines = lines[1:]
        random.shuffle(data_lines)
        eval_lines = data_lines[:N]
        for line in eval_lines:
            eval_file.write(line)

        print(f"Written Eval data to {eval_csv_path}")

        # Write remaining lines to train file
        with open(remaining_csv_path, "w", encoding="utf-8") as train_file:
            train_file.write(header)
            for line in data_lines[N:]:
                train_file.write(line)

if __name__ == "__main__":
    # Example usage
    # khmer_input_csv = "../khmer_ipa.csv"
    lao_input_csv = "../lao_ipa_cleaned.csv"

    # split_csv(khmer_input_csv, "khmer")
    split_csv(lao_input_csv, "lao")
            
