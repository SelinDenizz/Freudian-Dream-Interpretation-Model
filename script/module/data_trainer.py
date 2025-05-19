from unsloth import FastLanguageModel
import os
import json
import torch
from datasets import load_dataset
from transformers import TrainingArguments
 
class UnslothTrainer:
    def __init__(self,
                 model_name="meta-llama/Llama-2-7b-hf",
                 max_seq_length=2048,
                 micro_batch_size=1,
                 gradient_accumulation_steps=4,
                 num_epochs=3,
                 learning_rate=2e-4,
                 lora_r=16,
                 lora_alpha=32,
                 lora_dropout=0.05,
                 bf16=False,
                 tf32=False,
                 save_steps=100):
 
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.micro_batch_size = micro_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.bf16 = bf16
        self.tf32 = tf32
        self.save_steps = save_steps
 
    def finetune(self, jsonl_file, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
 
        format_name = self._detect_format(jsonl_file)
        dataset = load_dataset("json", data_files=jsonl_file, split="train")
        print(f"Loaded {len(dataset)} examples")
        print("Sample entry:", dataset[0])
 
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=torch.bfloat16 if self.bf16 else torch.float16,
            load_in_4bit=True,
        )
 
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            max_seq_length=self.max_seq_length,
            use_gradient_checkpointing=True,
            random_state=42,
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.micro_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            bf16=self.bf16,
            tf32=self.tf32,
            logging_steps=10,
            save_steps=self.save_steps,
            save_total_limit=3,
            report_to="tensorboard",
            push_to_hub=False,
        )
 
        if format_name in ["llama", "mistral"]:
            train_dataset = FastLanguageModel.get_dataset(dataset, tokenizer, self.max_seq_length, "text")
        else:
            def chatml_formatting_func(example):
                formatted = ""
                for m in example["messages"]:
                    role, content = m["role"], m["content"]
                    formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                return {"text": formatted}
 
            processed_dataset = dataset.map(chatml_formatting_func)
            train_dataset = FastLanguageModel.get_dataset(processed_dataset, tokenizer, self.max_seq_length, "text")
 
        trainer = FastLanguageModel.get_trainer(
            model=model,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            args=training_args,
        )
 
        trainer.train()
        trainer.save_model(os.path.join(output_dir, "final_model"))
 
        with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
            json.dump(self.__dict__ | {"format_name": format_name}, f, indent=2)
 
        print(f"Training complete. Model saved to {output_dir}")
        return os.path.join(output_dir, "final_model")
 
    def merge_lora_weights(self, model_path, output_dir):
        print("Merging LoRA weights with base model...")
        os.makedirs(output_dir, exist_ok=True)
 
        with open(os.path.join(os.path.dirname(model_path), "hyperparameters.json"), "r") as f:
            hparams = json.load(f)
 
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=hparams["model_name"],
            max_seq_length=hparams["max_seq_length"],
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
 
        model = FastLanguageModel.from_pretrained(model=model, model_name=model_path)
        model = FastLanguageModel.merge_lora_weights(model)
 
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Merged model saved to {output_dir}")
        return output_dir
 
    def _detect_format(self, filename):
        name = filename.lower()
        if "llama" in name:
            return "llama"
        elif "mistral" in name:
            return "mistral"
        return "chatml"