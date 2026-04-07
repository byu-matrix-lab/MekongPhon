from dict_transliterator import GreedyTransliterator
from clean import KhmerCleaner
from tqdm import tqdm

khmer_cleaner = KhmerCleaner()

def clean_khmer(sentence):
    # Returns true if the sentence is to be kept, false otherwise.
    if khmer_cleaner.check_unicode_ratio(sentence) < 0.65:
        return False

    return True

def read_mapping_csv(mapping_csv_path):
    mapping = {}

    with open(mapping_csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            orth, pron = line.strip().split(",")
            mapping[orth] = pron

    return mapping

def main():
    train_mapping_csv_path = "../data/khmer_train.csv"
    eval_mapping_csv_path = "../data/khmer_ipa.csv"
    source_data_path = "../data/opus/km-combined-dedup.txt"
    target_train_ipa_path = "../data/opus/km-train-ipa.txt"
    target_train_orth_path = "../data/opus/km-train-orth.txt"
    target_eval_ipa_path = "../data/opus/km-eval-ipa.txt"
    target_eval_orth_path = "../data/opus/km-eval-orth.txt"

    # Load mapping from csv
    train_mapping = read_mapping_csv(train_mapping_csv_path)
    eval_mapping = read_mapping_csv(eval_mapping_csv_path)
    
    # Init transliterators
    train_transliterator = GreedyTransliterator(train_mapping)
    eval_transliterator = GreedyTransliterator(eval_mapping)

    train_orth_sents = []
    train_ipa_sents = []
    eval_orth_sents = []
    eval_ipa_sents = []
    train_count = 0
    eval_count = 0

    # Read the input file
    with open(source_data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

        for line in tqdm(lines):
            # khmer_sent = line.strip().split("\t")[1]
            khmer_sent = line.strip()
            if not clean_khmer(khmer_sent):
                continue

            # Transliterate using both transliterators
            train_ipa_sent = train_transliterator.transliterate(khmer_sent)
            eval_ipa_sent = eval_transliterator.transliterate(khmer_sent)

            # Check if there are any khmer characters in the ipa sentences
            if khmer_cleaner.check_unicode_ratio(train_ipa_sent) == 0:
                train_orth_sents.append(khmer_sent)
                train_ipa_sents.append(train_ipa_sent)
                train_count += 1
            if khmer_cleaner.check_unicode_ratio(eval_ipa_sent) == 0:
                eval_orth_sents.append(khmer_sent)
                eval_ipa_sents.append(eval_ipa_sent)
                eval_count += 1

            
    print("Total final valid training sentences: ", train_count)
    print("Total final valid evaluation sentences: ", eval_count)

    # Filter out the orthgraphic sentences occuring in the train from the eval set
    filtered_eval_orth_sents = []
    filtered_eval_ipa_sents = []
    train_orth_set = set(train_orth_sents)
    for orth_sent, ipa_sent in zip(eval_orth_sents, eval_ipa_sents):
        if orth_sent not in train_orth_set:
            filtered_eval_orth_sents.append(orth_sent)
            filtered_eval_ipa_sents.append(ipa_sent)

    print("Total final eval sentences after removing train overlap: ", len(filtered_eval_orth_sents))

    # Write the output files
    with open(target_train_ipa_path, "w", encoding="utf-8") as f:
        for ipa_sent in train_ipa_sents:
            f.write(ipa_sent + "\n")
    with open(target_train_orth_path, "w", encoding="utf-8") as f:
        for orth_sent in train_orth_sents:
            f.write(orth_sent + "\n")
    with open(target_eval_ipa_path, "w", encoding="utf-8") as f:
        for ipa_sent in filtered_eval_ipa_sents:
            f.write(ipa_sent + "\n")
    with open(target_eval_orth_path, "w", encoding="utf-8") as f:
        for orth_sent in filtered_eval_orth_sents:
            f.write(orth_sent + "\n")
    
        
if __name__ == "__main__":
    main()