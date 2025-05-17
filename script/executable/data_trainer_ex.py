import argparse
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from script.module.data_trainer import UnslothTrainer  
 
def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM with Unsloth")
 
    # Main arguments
    parser.add_argument("--jsonl_file", type=str, required=True, help="Path to jsonl training data")
    parser.add_argument("--output_dir", type=str, default="unsloth_model", help="Directory to save output")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
 
    # Training arguments
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
 
    # LoRA config
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
 
    # Precision and saving
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--save_steps", type=int, default=100)
 
    # Merge LoRA after training
    parser.add_argument("--merge_weights", action="store_true")
    parser.add_argument("--merged_output_dir", type=str, default=None)
 
    args = parser.parse_args()
 
    # Init trainer
    trainer = UnslothTrainer(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bf16=args.bf16,
        tf32=args.tf32,
        save_steps=args.save_steps
    )
 
    # Train
    model_path = trainer.finetune(args.jsonl_file, args.output_dir)
 
    # Optionally merge LoRA weights
    if args.merge_weights:
        merged_path = args.merged_output_dir or f"{args.output_dir}/merged"
        trainer.merge_lora_weights(model_path, merged_path)
 
if __name__ == "__main__":
    main()