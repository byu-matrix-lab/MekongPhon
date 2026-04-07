# pipeline/

Scripts for building the MekongPhon corpus from scratch.

## Steps

1. **Get frequency lists** from http://sealang.net/project/list/ for Lao and Khmer. Clean the HTML and extract individual words into chunks (to avoid scraping the entire list at once).

2. **Scrape IPA** — run `scrape.py` on the chunked word lists. It fetches IPA for each word and records any failures.

3. **Clean the lexicon** — minor post-processing including converting Lao tone markings to IPA format. Manually add numerals and punctuation. This produces the final lexicons in `../data/lexicons/`.

4. **Download the text corpus** — use the scripts in `../data/opus_download/` to download OPUS parallel sentences for each language.

5. **Split the lexicon** — run `split_lexicon.py` to partition the lexicon into train and eval sets.

6. **Convert sentences to IPA** — use `lexicon_transliterator.py` (trie-based lookup) with `convert_lao.py` or `convert_khmer.py` to transliterate each sentence. Sentences with any out-of-lexicon tokens are filtered out. Output goes to `../data/splits/`.

7. **Use the corpus** — the resulting splits are ready for baseline evaluation or neural model training (see `../training/`).
