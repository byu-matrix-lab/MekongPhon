from dict_transliterator import GreedyTransliterator
from clean import LaoCleaner   # assumes you have a LaoCleaner like KhmerCleaner
from tqdm import tqdm

lao_cleaner = LaoCleaner()

def clean_lao(sentence):
    # Returns true if the sentence is to be kept, false otherwise.
    if lao_cleaner.check_unicode_ratio(sentence) < 0.65:
        return False
    return True

def read_mapping_csv(mapping_csv_path):
    mapping_tones = {}
    mapping_no_tones = {}

    with open(mapping_csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            orth, pron, tones = line.strip().split(",")
            mapping_tones[orth] = tones
            mapping_no_tones[orth] = pron

    return mapping_tones, mapping_no_tones

def main():
    # File paths
    train_mapping_csv_path = "../data/lao_train_cleaned.csv"
    eval_mapping_csv_path = "../data/lao_ipa_cleaned.csv"
    source_data_path = "../data/opus/lo-combined-dedup.txt"

    target_train_ipa_path = "../data/opus_cleaned/lo-train-ipa.txt"
    target_train_no_tones_path = "../data/opus_cleaned/lo-train-ipa-no-tones.txt"
    target_train_orth_path = "../data/opus_cleaned/lo-train-orth.txt"

    target_eval_ipa_path = "../data/opus_cleaned/lo-eval-ipa.txt"
    target_eval_no_tones_path = "../data/opus_cleaned/lo-eval-ipa-no-tones.txt"
    target_eval_orth_path = "../data/opus_cleaned/lo-eval-orth.txt"

    # Load mappings from CSVs
    train_mapping_tones, train_mapping_no_tones = read_mapping_csv(train_mapping_csv_path)
    eval_mapping_tones, eval_mapping_no_tones = read_mapping_csv(eval_mapping_csv_path)
    
    # Initialize transliterators for both tone and no-tone data
    train_trans_tones = GreedyTransliterator(train_mapping_tones)
    train_trans_no_tones = GreedyTransliterator(train_mapping_no_tones)
    eval_trans_tones = GreedyTransliterator(eval_mapping_tones)
    eval_trans_no_tones = GreedyTransliterator(eval_mapping_no_tones)

    # Containers
    train_orth_sents, train_ipa_sents, train_ipa_no_tones_sents = [], [], []
    eval_orth_sents, eval_ipa_sents, eval_ipa_no_tones_sents = [], [], []
    train_count, eval_count = 0, 0

    # Read the source sentences
    with open(source_data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            lao_sent = line.strip()
            if not clean_lao(lao_sent):
                continue

            # Transliterate both tone and no-tone forms
            train_ipa = train_trans_tones.transliterate(lao_sent)
            train_ipa_no_tones = train_trans_no_tones.transliterate(lao_sent)
            eval_ipa = eval_trans_tones.transliterate(lao_sent)
            eval_ipa_no_tones = eval_trans_no_tones.transliterate(lao_sent)

            # Check for successful transliteration (no Lao chars left)
            if lao_cleaner.check_unicode_ratio(train_ipa) == 0:
                train_orth_sents.append(lao_sent)
                train_ipa_sents.append(train_ipa)
                train_ipa_no_tones_sents.append(train_ipa_no_tones)
                train_count += 1
            if lao_cleaner.check_unicode_ratio(eval_ipa) == 0:
                eval_orth_sents.append(lao_sent)
                eval_ipa_sents.append(eval_ipa)
                eval_ipa_no_tones_sents.append(eval_ipa_no_tones)
                eval_count += 1

    print("Total final valid training sentences:", train_count)
    print("Total final valid evaluation sentences:", eval_count)

    # Remove overlaps
    filtered_eval_orth, filtered_eval_ipa, filtered_eval_no_tones = [], [], []
    train_orth_set = set(train_orth_sents)
    for orth, ipa, ipa_no_tones in zip(eval_orth_sents, eval_ipa_sents, eval_ipa_no_tones_sents):
        if orth not in train_orth_set:
            filtered_eval_orth.append(orth)
            filtered_eval_ipa.append(ipa)
            filtered_eval_no_tones.append(ipa_no_tones)

    print("Final eval sentences after removing overlaps:", len(filtered_eval_orth))

    # Write outputs
    def write_list(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(item + "\n")

    write_list(target_train_ipa_path, train_ipa_sents)
    write_list(target_train_no_tones_path, train_ipa_no_tones_sents)
    write_list(target_train_orth_path, train_orth_sents)

    write_list(target_eval_ipa_path, filtered_eval_ipa)
    write_list(target_eval_no_tones_path, filtered_eval_no_tones)
    write_list(target_eval_orth_path, filtered_eval_orth)


if __name__ == "__main__":
    main()
