import epitran
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os


def transliterate_lines(epi, lines):
    """Transliterate a list of text lines using a preloaded Epitran instance."""
    results = []
    for line in lines:
        line = line.strip()
        if not line:
            results.append("")
            continue
        results.append(epi.transliterate(line))
    return results


def main():
    parser = argparse.ArgumentParser(description="Transliterate text to IPA using Epitran.")
    parser.add_argument("language_code", type=str, help="Language code for Epitran (e.g., 'tha-Thai', 'khm-Khmr', 'lao-Laoo').")
    parser.add_argument("text_file", type=str, help="Text file with one line per input.")
    parser.add_argument("output_file", type=str, help="File to save the IPA transliteration.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel threads to use.")
    parser.add_argument("--chunk_size", type=int, default=200, help="Number of lines per thread batch.")
    args = parser.parse_args()

    # Read all input lines, preserving blank lines for alignment
    with open(args.text_file, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    # Split input into chunks for each thread
    chunks = [lines[i:i + args.chunk_size] for i in range(0, len(lines), args.chunk_size)]

    # We'll create one Epitran instance per thread
    def process_chunk(chunk):
        epi = epitran.Epitran(args.language_code)
        return transliterate_lines(epi, chunk)

    # Use executor.map() to preserve chunk order
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        chunk_results = list(tqdm(
            executor.map(process_chunk, chunks),
            total=len(chunks),
            desc="Transliterating"
        ))

    for chunk_result in chunk_results:
        results.extend(chunk_result)

    # Save results
    with open(args.output_file, "w", encoding="utf-8") as out_f:
        out_f.write("\n".join(results) + "\n")


if __name__ == "__main__":
    main()