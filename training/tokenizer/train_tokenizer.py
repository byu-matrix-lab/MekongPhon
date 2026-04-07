import argparse
import os
import sys
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel, UnicodeScripts
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tqdm import tqdm
import time
import random

def train(tok_path, vocab_size, texts, special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]", "[SEP]"]):
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Pre-tokenizer splits text into words by chars
    # tokenizer.pre_tokenizer = UnicodeScripts()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

    # Attach ByteLevel decoder
    tokenizer.decoder = ByteLevelDecoder()


    # Configure training parameters
    trainer = BpeTrainer(
        vocab_size=vocab_size,            # your desired vocab size
        special_tokens=special_tokens,
        min_frequency=2             # minimum occurrences of a pair to be merged
    )

    # Train on list of text files or iterable of strings
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Check if the directory exists, if not create it
    directory = os.path.dirname(tok_path)

    if not os.path.exists(directory) and directory != "":
        os.makedirs(directory)

    # Save the tokenizer to the specified path
    tokenizer.save(tok_path)
    

def main():
    parser = argparse.ArgumentParser(
        description="Train a tokenizer on one or more text files."
    )
    parser.add_argument(
        "tokenizer_path",
        help="Path where the tokenizer will be saved."
    )
    parser.add_argument(
        "vocab_size",
        type=int,
        help="Vocabulary size for the tokenizer."
    )
    parser.add_argument(
        "text_files",
        nargs="+",
        help="One or more text files to train on."
    )
    parser.add_argument(
        "--special-tokens",
        default="[UNK],[PAD],[BOS],[EOS]",
        help="Comma-separated list of special tokens (default: [UNK],[PAD],[BOS],[EOS])"
    )

    args = parser.parse_args()

    tok_path = args.tokenizer_path
    vocab_size = args.vocab_size
    special_tokens = args.special_tokens.split(',')
    text_files = args.text_files  # always a list, even if only one file

    # args = sys.argv[1:]
    # if len(args) < 3:
    #     print("Usage: python train_tokenizer.py <tokenizer_path> <vocab_size> <special_tokens> <text_file1> <text_file2> ...")
    #     sys.exit(1)
    
    # tok_path = args[0]
    # vocab_size = int(args[1])
    # special_tokens = args[2].split(',')
    # text_files = args[3:]

    # Ensure it's always a list, even if a single string is passed
    if isinstance(text_files, str):
        text_files = [text_files]
    
    texts = []

    for path in tqdm(text_files, desc="Reading text files", unit="file"):
        with open(path, 'r', encoding='utf-8') as f:
            # Read the file and split it into lines
            lines = f.readlines()

            # Remove leading and trailing whitespace from each line
            lines = [line.strip() for line in lines]

            # Append the lines to the texts list, no need to shuffle because Huggingface does frequency globaly
            texts.append(lines)

    # Downsample to smallest file
    min_length = min(len(text) for text in texts)
    # Shuffle and downsample each text to the minimum length
    for i in range(len(texts)):
        if len(texts[i]) > min_length:
            # Shuffle
            random.shuffle(texts[i])
            texts[i] = texts[i][:min_length]
    
    # Flatten the list of lists into a single list
    texts = [line for sublist in texts for line in sublist]

    print("Starting training tokenizer...")

    start_time = time.time()

    # Train the tokenizer
    train(tok_path, vocab_size, texts, special_tokens=special_tokens)

    end_time = time.time()
    total_time = end_time - start_time

    # Print the min:sec format
    print(f"Tokenizer trained and saved to {tok_path} in {int(total_time // 60)}m {int(total_time % 60)}s")

if __name__ == "__main__":
    main()