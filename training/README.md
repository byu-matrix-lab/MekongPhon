# training/

Code for training and evaluating IPA transliteration models.

## Subdirectories

- `tokenizer/` — BPE tokenizer training
- `model/` — seq2seq (BERT encoder-decoder) training
- `baselines/` — rule-based and Epitran baselines
- `evaluation/` — scoring scripts (BLEU, chrF++, CER)

## Workflow

### 1. Train tokenizers

Train separate BPE tokenizers for the source (script) and target (IPA) sides:

```bash
python tokenizer/train_tokenizer.py <output_path> <vocab_size> <text_file1> [<text_file2> ...]
```

### 2. Train the seq2seq model

```bash
python model/train.py \
  --train_source_file <src_train> \
  --train_target_file <tgt_train> \
  --val_source_file <src_val> \
  --val_target_file <tgt_val> \
  --encoder_tokenizer_file <encoder.json> \
  --decoder_tokenizer_file <decoder.json> \
  --output_dir <output_dir>
```

Key optional arguments: `--hidden_size`, `--num_layers`, `--num_heads`, `--epochs`, `--lr`. Run with `--help` for the full list.

### 3. Evaluate a trained model

Generate hypotheses and score them:

```bash
python evaluation/generate_and_eval.py \
  --model_path <model_dir> \
  --source_file <src_test> \
  --reference_file <tgt_test> \
  --output_file generated.txt
```

To score pre-generated hypotheses:

```bash
python evaluation/eval.py \
  --hypothesis_file generated.txt \
  --reference_file <tgt_test>
```

Reports BLEU, chrF++, and CER.

## Baselines

Run Epitran on a text file:

```bash
python baselines/epitran_inference.py <language_code> <input_file> <output_file>
```

Example language codes: `khm-Khmr`, `lao-Laoo`.

Rule-based baselines are in `baselines/rb_khm_to_ipa.py` and `baselines/rb_lao_to_ipa.py`.
