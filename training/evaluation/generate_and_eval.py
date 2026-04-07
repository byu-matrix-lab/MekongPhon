import argparse
import torch
from tqdm import tqdm
import sacrebleu
import evaluate
from transformers import EncoderDecoderModel, PreTrainedTokenizerFast

def generate_translations(model, encoder_tokenizer, decoder_tokenizer, source_sents, batch_size=16, device="cpu"):
    """
    Generates translations for a list of source sentences.
    """
    model.to(device)
    model.eval()

    hypotheses = []
    print(f"Generating translations with batch size {batch_size}...")
    for i in tqdm(range(0, len(source_sents), batch_size), desc="Translating"):
        batch = source_sents[i:i + batch_size]
        
        # Tokenize the batch
        inputs = encoder_tokenizer(
            batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(device)

        # Generate translations
        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode and add to the list
        decoded_preds = decoder_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        hypotheses.extend(decoded_preds)
        
    return hypotheses

def calculate_scores(hypotheses, references):
    """
    Calculates and returns BLEU, chrF++, and CER scores.
    """
    print("\nCalculating scores...")
    
    # 1. BLEU Score (using sacrebleu)
    bleu_metric = evaluate.load("sacrebleu")
    # Sacrebleu requires references to be a list of lists
    bleu_score = bleu_metric.compute(predictions=hypotheses, references=[[ref] for ref in references])

    # 2. chrF++ Score
    chrf_metric = evaluate.load("chrf")
    # The word_order=2 argument enables the "++" variant
    chrf_score = chrf_metric.compute(predictions=hypotheses, references=references, word_order=2)

    # 3. Character Error Rate (CER)
    cer_metric = evaluate.load("cer")
    cer_score = cer_metric.compute(predictions=hypotheses, references=references)

    return {
        "sacrebleu": bleu_score,
        "chrf++": chrf_score,
        "cer": cer_score
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained seq2seq model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory.")
    parser.add_argument("--source_file", type=str, required=True, help="Path to the source text file (one sentence per line).")
    parser.add_argument("--reference_file", type=str, required=True, help="Path to the reference/target text file (one sentence per line).")
    parser.add_argument("--output_file", type=str, default="generated.txt", help="File to save the generated translations.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation.")
    
    args = parser.parse_args()

    # --- 1. Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 2. Load Model and Tokenizers ---
    print(f"Loading model and tokenizers from {args.model_path}...")
    model = EncoderDecoderModel.from_pretrained(args.model_path)
    encoder_tokenizer = PreTrainedTokenizerFast.from_pretrained(f"{args.model_path}/encoder_tokenizer")
    decoder_tokenizer = PreTrainedTokenizerFast.from_pretrained(f"{args.model_path}/decoder_tokenizer")

    # --- 3. Load Data ---
    with open(args.source_file, "r", encoding="utf-8") as f:
        source_sents = [line.strip() for line in f.readlines()]
    with open(args.reference_file, "r", encoding="utf-8") as f:
        reference_sents = [line.strip() for line in f.readlines()]

    if len(source_sents) != len(reference_sents):
        raise ValueError("Source and reference files must have the same number of lines.")
    
    print(f"Loaded {len(source_sents)} sentences for evaluation.")

    # --- 4. Generate Hypotheses ---
    hypotheses = generate_translations(model, encoder_tokenizer, decoder_tokenizer, source_sents, args.batch_size, device)

    # --- 5. Calculate Scores ---
    scores = calculate_scores(hypotheses, reference_sents)

    # --- 6. Print Results ---
    print("\n" + "="*25)
    print("      EVALUATION RESULTS")
    print("="*25)
    print(f"  BLEU Score    : {scores['sacrebleu']['score']:.2f}")
    print(f"  chrF++ Score  : {scores['chrf++']['score']:.2f}")
    print(f"  CER Score     : {scores['cer']:.4f} (Lower is better)")
    print("="*25)
    
    # Print a few examples
    print("\n--- Sample Generations ---")
    for i in range(min(5, len(source_sents))):
        print(f"Source:    '{source_sents[i]}'")
        print(f"Reference: '{reference_sents[i]}'")
        print(f"Generated: '{hypotheses[i]}'")
        print("-" * 20)

    # --- 7. Save Generated Outputs ---
    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for line in hypotheses:
            out_f.write(line + "\n")

    print(f"\nGenerated translations saved to {args.output_file}")


if __name__ == "__main__":
    main()