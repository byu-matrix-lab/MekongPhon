import argparse
import os
import torch
from transformers import (
    EncoderDecoderModel,
    AutoConfig,
    BertModel,
    BertLMHeadModel,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerFast,
    TrainerCallback,
    EarlyStoppingCallback,
)
from tokenizers import Tokenizer
from datasets import Dataset

class GenerationCallback(TrainerCallback):
    """
    A custom callback to generate and print a few validation examples during evaluation.
    """
    def __init__(self, val_dataset, encoder_tokenizer, decoder_tokenizer, num_examples=5, num_beams=4):
        self.val_dataset = val_dataset
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.num_examples = num_examples
        self.num_beams = num_beams

    def on_evaluate(self, args, state, control, model, **kwargs):
        # Select a few examples to generate from
        sample_dataset = self.val_dataset.select(range(self.num_examples))
        print("\n" + "="*50)
        print(f"GENERATION EXAMPLES AT STEP {state.global_step}")
        print("="*50)

        for example in sample_dataset:
            source_text = example["input_texts"]
            target_text = example["target_texts"]

            # Tokenize the source text and move to the correct device
            inputs = self.encoder_tokenizer(
                source_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.encoder_tokenizer.model_max_length
            ).to(model.device)

            # Generate output tokens
            generated_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=self.decoder_tokenizer.model_max_length,
                num_beams=self.num_beams,
                early_stopping=True
            )

            # Decode the generated tokens back to text
            generated_text = self.decoder_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            print(f"  Input:    '{source_text}'")
            print(f"  Target:   '{target_text}'")
            print(f"  Generated:  '{generated_text}'")
            print("-" * 20)
        print("="*50 + "\n")

def main(args):
    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

    # Custom tokenizers
    print("Loading tokenizers...")
    encoder_tokenizer_obj = Tokenizer.from_file(args.encoder_tokenizer_file)
    decoder_tokenizer_obj = Tokenizer.from_file(args.decoder_tokenizer_file)

    encoder_tokenizer = PreTrainedTokenizerFast(tokenizer_object=encoder_tokenizer_obj)
    decoder_tokenizer = PreTrainedTokenizerFast(tokenizer_object=decoder_tokenizer_obj)

    encoder_tokenizer.pad_token = "[PAD]"
    decoder_tokenizer.pad_token = "[PAD]"
    encoder_tokenizer.model_max_length = args.max_len
    decoder_tokenizer.model_max_length = args.max_len

    # Add BOS/EOS tokens if not present, necessary for generation
    if decoder_tokenizer.bos_token is None:
        decoder_tokenizer.bos_token = "[BOS]"
    if decoder_tokenizer.eos_token is None:
        decoder_tokenizer.eos_token = "[EOS]"

    # Read data
    print("Loading and preparing datasets...")
    with open(args.train_source_file, "r", encoding="utf-8") as f:
        train_input_texts = [line.strip() for line in f.readlines()]
    with open(args.train_target_file, "r", encoding="utf-8") as f:
        train_target_texts = [line.strip() for line in f.readlines()]
    train_dataset = Dataset.from_dict({"input_texts": train_input_texts, "target_texts": train_target_texts})

    with open(args.val_source_file, "r", encoding="utf-8") as f:
        val_input_texts = [line.strip() for line in f.readlines()]
    with open(args.val_target_file, "r", encoding="utf-8") as f:
        val_target_texts = [line.strip() for line in f.readlines()]
    val_dataset = Dataset.from_dict({"input_texts": val_input_texts, "target_texts": val_target_texts})

    # Preprocess function
    def preprocess_function(examples):
        srcs = [text for text in examples["input_texts"]]
        tgts = [f"{decoder_tokenizer.bos_token}{text}{decoder_tokenizer.eos_token}" for text in examples["target_texts"]]

        input_encodings = encoder_tokenizer(srcs, truncation=True, padding="max_length", max_length=args.max_len)
        target_encodings = decoder_tokenizer(tgts, truncation=True, padding="max_length", max_length=args.max_len)

        labels = [
            [(tok_id if tok_id != decoder_tokenizer.pad_token_id else -100) for tok_id in ids]
            for ids in target_encodings["input_ids"]
        ]
        return {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": labels,
        }

    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

    # Configure and init model
    print("Initializing model from scratch...")
    encoder_config = AutoConfig.from_pretrained(
        "bert-base-uncased", 
        vocab_size=encoder_tokenizer.vocab_size, 
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_layers, 
        num_attention_heads=args.num_heads, 
        intermediate_size=args.intermediate_size,
        # local_files_only=True
    )
    decoder_config = AutoConfig.from_pretrained(
        "bert-base-uncased", 
        vocab_size=decoder_tokenizer.vocab_size, 
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_layers, 
        num_attention_heads=args.num_heads, 
        intermediate_size=args.intermediate_size, 
        is_decoder=True, 
        add_cross_attention=True, 
        bos_token_id=decoder_tokenizer.bos_token_id, 
        eos_token_id=decoder_tokenizer.eos_token_id, 
        pad_token_id=decoder_tokenizer.pad_token_id,
        # local_files_only=True
    )

    encoder_model = BertModel(encoder_config)
    decoder_model = BertLMHeadModel(decoder_config)
    model = EncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)

    model.config.decoder_start_token_id = decoder_tokenizer.bos_token_id
    model.config.eos_token_id = decoder_tokenizer.eos_token_id
    model.config.pad_token_id = decoder_tokenizer.pad_token_id
    model.config.vocab_size = decoder_tokenizer.vocab_size

    # Training args and callbacks
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Instantiate callbacks
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=args.patience)
    generation_callback = GenerationCallback(
        val_dataset, 
        encoder_tokenizer, 
        decoder_tokenizer, 
        num_examples=args.num_gen_examples,
        num_beams=args.num_beams
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        callbacks=[early_stopping_callback, generation_callback],
    )

    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Training complete! Saving best model and tokenizers to '{args.output_dir}'.")
    model.save_pretrained(args.output_dir)
    encoder_tokenizer.save_pretrained(os.path.join(args.output_dir, "encoder_tokenizer"))
    decoder_tokenizer.save_pretrained(os.path.join(args.output_dir, "decoder_tokenizer"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Sequence-to-Sequence model from scratch.")

    # --- File Path Arguments ---
    path_group = parser.add_argument_group("File Paths")
    path_group.add_argument("--train_source_file", type=str, required=True, help="Path to the training source file.")
    path_group.add_argument("--train_target_file", type=str, required=True, help="Path to the training target file.")
    path_group.add_argument("--val_source_file", type=str, required=True, help="Path to the validation source file.")
    path_group.add_argument("--val_target_file", type=str, required=True, help="Path to the validation target file.")
    path_group.add_argument("--encoder_tokenizer_file", type=str, required=True, help="Path to the trained encoder tokenizer file (.json).")
    path_group.add_argument("--decoder_tokenizer_file", type=str, required=True, help="Path to the trained decoder tokenizer file (.json).")
    path_group.add_argument("--output_dir", "-o", type=str, default="./trained_seq2seq_model", help="Directory to save the final model and checkpoints.")

    # --- Model Hyperparameter Arguments ---
    model_group = parser.add_argument_group("Model Hyperparameters")
    model_group.add_argument("--max_len", type=int, default=128, help="Maximum sequence length for tokenization.")
    model_group.add_argument("--hidden_size", type=int, default=256, help="Model hidden size.")
    model_group.add_argument("--num_layers", type=int, default=3, help="Number of layers in the encoder and decoder.")
    model_group.add_argument("--num_heads", type=int, default=4, help="Number of attention heads.")
    model_group.add_argument("--intermediate_size", type=int, default=512, help="Size of the feed-forward layer.")

    # --- Training Hyperparameter Arguments ---
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    train_group.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training.")
    train_group.add_argument("--eval_batch_size", type=int, default=4, help="Batch size for evaluation.")
    train_group.add_argument("--lr", "--learning_rate", type=float, default=5e-4, help="Learning rate.")
    train_group.add_argument("--patience", type=int, default=3, help="Patience for early stopping (number of evaluations with no improvement).")
    train_group.add_argument("--eval_steps", type=int, default=2000, help="Run evaluation every N steps.")
    train_group.add_argument("--save_steps", type=int, default=2000, help="Save a checkpoint every N steps.")
    train_group.add_argument("--logging_steps", type=int, default=500, help="Log training information every N steps.")
    train_group.add_argument("--save_total_limit", type=int, default=2, help="Maximum number of checkpoints to keep.")

    # --- Generation Callback Arguments ---
    gen_group = parser.add_argument_group("Generation Callback Settings")
    gen_group.add_argument("--num_gen_examples", type=int, default=5, help="Number of examples to generate during evaluation.")
    gen_group.add_argument("--num_beams", type=int, default=4, help="Number of beams for beam search generation.")
    
    args = parser.parse_args()
    main(args)