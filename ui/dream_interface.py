import os
import sys
import json
import torch
import logging
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import spaces

class DreamAnalyzer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.model_format = "llama"
        self.load_error = None
        
        try:
            self._load_model()
        except Exception as e:
            self.load_error = str(e)
            print(f"ERROR: Failed to load model: {e}")

    def _load_model(self):
        print(f"Loading model from {self.model_path}...")

        if os.path.exists(self.model_path):
            print(f"Files in {self.model_path}:")
            for file in os.listdir(self.model_path):
                print(f"  - {file}")
        else:
            print(f"WARNING: Model path {self.model_path} does not exist!")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        adapter_config_path = os.path.join(self.model_path, "adapter_config.json")
        is_lora = os.path.exists(adapter_config_path)

        if is_lora:
            print(f"Found adapter_config.json - treating as LoRA model")
            
            # Try to find hyperparameters for base model info
            hyperparameters_path = os.path.join(os.path.dirname(self.model_path), "hyperparameters.json")
            if os.path.exists(hyperparameters_path):
                with open(hyperparameters_path, "r") as f:
                    hyperparameters = json.load(f)
                    self.model_format = hyperparameters.get("format_name", "llama")
                    model_name = hyperparameters.get("model_name", "meta-llama/Llama-2-7b-chat-hf")
                    print(f"Found hyperparameters.json: format={self.model_format}, base_model={model_name}")
            else:
                model_name = "meta-llama/Llama-2-7b-chat-hf"
                print(f"No hyperparameters.json found, using default base model: {model_name}")

            print(f"Loading tokenizer from base model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"Loading base model: {model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

            from peft import PeftModel
            print(f"Loading LoRA adapter from: {self.model_path}")
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
            print("LoRA adapter loaded successfully")
        else:
            print(f"No adapter_config.json found - treating as full model")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

        print("Model loading completed successfully")

    @spaces.GPU  # ZeroGPU decorator for GPU allocation
    def analyze_dream(self, dream_text):
        print(f"[DEBUG] analyze_dream called with text: '{dream_text[:50]}...' (length: {len(dream_text)})")
        
        if self.load_error:
            return f"Model loading failed: {self.load_error}"
            
        if not self.model or not self.tokenizer:
            return "Model not loaded properly. Please check the logs."
            
        if not dream_text.strip():
            return "Please enter a dream to analyze."

        print("[DEBUG] Starting dream analysis with ZeroGPU...")
        
        if torch.cuda.is_available():
            print(f"[DEBUG] GPU available: {torch.cuda.get_device_name()}")
            print(f"[DEBUG] Memory before: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
            torch.cuda.empty_cache()

        if self.model_format == "llama":
            prompt = f"[INST] <<SYS>>\nYou are a Freudian dream analyst who provides insightful interpretations of dreams based on Freudian theory.\n<</SYS>>\n\nPlease analyze this dream from a Freudian perspective:\n\n{dream_text} [/INST]"
        elif self.model_format == "mistral":
            prompt = f"[INST] Please analyze this dream from a Freudian perspective:\n\n{dream_text} [/INST]"
        else:
            prompt = f"<|im_start|>system\nYou are a Freudian dream analyst.<|im_end|>\n<|im_start|>user\nAnalyze this dream:\n\n{dream_text}<|im_end|>\n<|im_start|>assistant\n"

        print(f"[DEBUG] Using prompt format: {self.model_format}")
        print(f"[DEBUG] Prompt length: {len(prompt)}")

        try:
            print("[DEBUG] Tokenizing input...")
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1200)
            print(f"[DEBUG] Input shape: {inputs['input_ids'].shape}")
            
            print("[DEBUG] Starting generation...")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,  # Reduced for ZeroGPU timeout
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            print(f"[DEBUG] Generation completed. Output shape: {outputs.shape}")
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"[DEBUG] Full response length: {len(full_response)}")
            
            analysis_text = full_response[len(prompt):].strip()
            print(f"[DEBUG] Extracted analysis length: {len(analysis_text)}")
            print(f"[DEBUG] Analysis preview: {analysis_text[:100]}...")
            
            if not analysis_text.strip():
                return "⚠️ The model generated an empty response. Please try a different dream description."
            
            formatted = analysis_text.replace("**", "").replace("[/INST]", "").strip()
            
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"[DEBUG] Memory after: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

            print("[DEBUG] Dream analysis completed successfully")
            
            # Return formatted HTML
            return f"""
            <div style='padding: 1.5em; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #2563eb;'>
                <h3 style='color: #2563eb; margin-top: 0;'>🧠 Freudian Dream Analysis</h3>
                <div style='line-height: 1.6; color: #333;'>{formatted}</div>
            </div>
            """

        except Exception as e:
            print(f"[DEBUG] Error during analysis: {e}")
            print(f"[DEBUG] Error type: {type(e).__name__}")
            import traceback
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            return f"Analysis failed: {str(e)}"

    def record_feedback(self, dream_text, analysis, rating, suggestion):
        try:
            os.makedirs("feedback_data", exist_ok=True)
            feedback = {
                "timestamp": datetime.now().isoformat(),
                "dream_text": dream_text,
                "rating": rating,
                "suggestion": suggestion
            }
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"feedback_data/feedback_{timestamp}.json", "w") as f:
                json.dump(feedback, f, indent=2)
            return "Thank you for your feedback!"
        except:
            return "Feedback saved locally."

def create_dream_interface(model_path):
    analyzer = DreamAnalyzer(model_path)
    
    with gr.Blocks(title="Freudian Dream Analyzer") as interface:
        gr.Markdown("#Freudian Dream Analyzer")
        gr.Markdown(f"""
        **Freudian psychoanalytic dream interpretation**
        
        Status: {'GPU' if torch.cuda.is_available() else '💻 CPU'}
        
        *For educational and entertainment purposes only.*
        """)
        
        with gr.Row():
            with gr.Column():
                dream_input = gr.Textbox(
                    lines=8,
                    placeholder="Describe your dream in detail...",
                    label="Dream Description"
                )
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze Dream", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    
        with gr.Row():
            with gr.Column():
                analysis_output = gr.HTML(label="Analysis")
                
        with gr.Accordion("Feedback", open=False):
            feedback_rating = gr.Slider(1, 5, 3, label="Rating")
            feedback_text = gr.Textbox(lines=2, label="Comments")
            feedback_btn = gr.Button("Submit Feedback")
            feedback_output = gr.Textbox(label="", visible=False)

        # Example dreams
        examples = [
            ["I was flying over water but suddenly started falling."],
            ["I was in my childhood home but the rooms were different."],
            ["I was being chased through a dark forest."],
            ["I was taking a test but couldn't understand the questions."],
            ["I was at a party but realized I wasn't wearing clothes."]
        ]
        
        gr.Examples(examples, inputs=dream_input)
        
        def clear_inputs():
            return "", ""
            
        analyze_btn.click(
            analyzer.analyze_dream,
            inputs=dream_input,
            outputs=analysis_output
        )
        
        clear_btn.click(
            clear_inputs,
            outputs=[dream_input, analysis_output]
        )
        
        feedback_btn.click(
            analyzer.record_feedback,
            inputs=[dream_input, analysis_output, feedback_rating, feedback_text],
            outputs=feedback_output
        )
        
    return interface