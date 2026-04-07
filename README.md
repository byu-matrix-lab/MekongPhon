# MekongPhon

A large-scale parallel IPA corpus for Lao and Khmer, along with training code for seq2seq transliteration models.

## Directories

- `data/` — lexicons, OPUS download scripts, train/eval splits, and qualitative samples
- `pipeline/` — scripts for building the corpus: scraping, lexicon construction, and sentence conversion
- `training/` — tokenizer training, seq2seq model training, baselines, and evaluation
- `models/` — links to published models on HuggingFace

## Citation

If you use this work, please cite (temporary citation before publication in May 2026):

```
@inproceedings{shurtz-etal-2026-mekongphon,
    title = "MekongPhon: A Large-Scale Parallel IPA Corpus for Lao and Khmer",
    author = "Shurtz, Ammon  and
              Richardson, Christian  and
              Richardson, Stephen D.",
    booktitle = "Proceedings of the Fifteenth biennial Language Resources and Evaluation Conference (LREC 2026)",
    year = "2026",
    address = "XXXX",
    publisher = "ELRA",
    pages = "XXXX--XXXX",
}
```
