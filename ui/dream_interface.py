import os
import sys
import json
import torch
import logging
import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

class DreamAnalyzer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.pipe, self.model_format = self._load_model()
        
    def _load_model(self):
        print(f"Loading model from {self.model_path}...")
        
        # Check if this is a LoRA model or full model
        is_lora = os.path.exists(os.path.join(os.path.dirname(self.model_path), "hyperparameters.json"))
        model_format = "llama"  
        
        if is_lora:
            with open(os.path.join(os.path.dirname(self.model_path), "hyperparameters.json"), "r") as f:
                hyperparameters = json.load(f)
                model_format = hyperparameters.get("format_name", "llama")
            
            model_name = hyperparameters.get("model_name", "NousResearch/Llama-2-7b-chat-hf")
            
            # For inference, we'll use regular HF model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
        else:
            # Load the full model directly
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
        
       
        try:
            text_generation_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
            )
            logging.info("Text generation pipeline created successfully")
        except Exception as e:
            logging.error("Failed to create text generation pipeline: {}".format(e))
        
        return text_generation_pipeline, model_format
    
    def analyze_dream(self, dream_text):
       
        if not dream_text.strip():
            return "Please enter a dream to analyze."
        
        if self.model_format == "llama":
            prompt = f"[INST] <<SYS>>\nYou are a Freudian dream analyst who provides insightful interpretations of dreams based on Freudian theory.\n<</SYS>>\n\nPlease analyze this dream from a Freudian perspective:\n\n{dream_text} [/INST]"
        elif self.model_format == "mistral":
            prompt = f"[INST] Please analyze this dream from a Freudian perspective:\n\n{dream_text} [/INST]"
        else:  # chatml
            prompt = f"<|im_start|>system\nYou are a Freudian dream analyst who provides insightful interpretations of dreams based on Freudian theory.<|im_end|>\n<|im_start|>user\nPlease analyze this dream from a Freudian perspective:\n\n{dream_text}<|im_end|>\n<|im_start|>assistant\n"
        
        result = self.pipe(prompt, return_full_text=False)
        
        analysis_text = result[0]["generated_text"]
        
        formatted_analysis = self._format_analysis(analysis_text)
        
        return formatted_analysis
    
    def _format_analysis(self, analysis_text):
        
        formatted = analysis_text
        
        chunks = []
        in_bold = False
        for i, part in enumerate(formatted.split('**')):
            if i % 2 == 0:  # Not bold
                chunks.append(part)
            else:  # Bold
                chunks.append(f"<b>{part}</b>")
        
        formatted = "".join(chunks)
        
        for section in ["Dream Analysis", "Manifest Content", "Potential Freudian Symbols", 
                       "Latent Content Analysis", "Possible Psychological Mechanisms"]:
            formatted = formatted.replace(f"<b>{section}</b>", f"<h3>{section}</h3>")
        
        lines = formatted.split('\n')
        formatted_lines = []
        in_list = False
        
        for line in lines:
            if line.strip().startswith("- "):
                if not in_list:
                    formatted_lines.append("<ul>")
                    in_list = True
                formatted_lines.append(f"<li>{line.strip()[2:]}</li>")
            else:
                if in_list:
                    formatted_lines.append("</ul>")
                    in_list = False
                formatted_lines.append(line)
        
        if in_list:
            formatted_lines.append("</ul>")
        
        formatted = "\n".join(formatted_lines)
        return formatted
    
    def record_feedback(self, dream_text, analysis, rating, improvement_suggestion, feedback_dir="feedback_data"):
        os.makedirs(feedback_dir, exist_ok=True)
        
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "dream_text": dream_text,
            "model_analysis": analysis,
            "rating": rating,
            "improvement_suggestion": improvement_suggestion
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        feedback_file = os.path.join(feedback_dir, f"feedback_{timestamp}.json")
        
        with open(feedback_file, "w") as f:
            json.dump(feedback, f, indent=2)
        
        master_file = os.path.join(feedback_dir, "all_feedback.json")
        
        if os.path.exists(master_file):
            with open(master_file, "r") as f:
                try:
                    all_feedback = json.load(f)
                except json.JSONDecodeError:
                    all_feedback = []
        else:
            all_feedback = []
        
        all_feedback.append(feedback)
        
        with open(master_file, "w") as f:
            json.dump(all_feedback, f, indent=2)
        
        return "Thank you for your feedback! It will help improve future versions of the dream analyzer."

def create_dream_interface(model_path):
    analyzer = DreamAnalyzer(model_path)
    
    def analyze_dream_fn(dream_text):
        return analyzer.analyze_dream(dream_text)
    
    def submit_feedback(dream_text, analysis, rating, suggestion):
        return analyzer.record_feedback(dream_text, analysis, rating, suggestion)
    
    with gr.Blocks(title="Freudian Dream Analyzer", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# Freudian Dream Analyzer")
        gr.Markdown("""
        This tool analyzes dreams from a Freudian psychoanalytic perspective. 
        Enter the description of a dream, and the model will provide an interpretation 
        based on Freudian theory of manifest content, latent content, and symbolic meanings.
        
        *Note: Dream analysis is subjective and this tool provides one possible interpretation 
        based on Freudian psychoanalysis.*
        """)
        
        # Input section
        with gr.Row():
            with gr.Column(scale=2):
                dream_input = gr.Textbox(
                    lines=10, 
                    placeholder="Enter your dream here...",
                    label="Dream Description"
                )
                analyze_btn = gr.Button("Analyze Dream", variant="primary")
                
        # Output section
        with gr.Row():
            with gr.Column(scale=2):
                analysis_output = gr.HTML(
                    label="Freudian Analysis",
                    elem_id="analysis_output"
                )
        
        # Feedback section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Help improve the Dream Analyzer")
                feedback_rating = gr.Slider(
                    minimum=1, 
                    maximum=5, 
                    step=1, 
                    value=3, 
                    label="Rate the quality of this analysis (1-5)"
                )
                feedback_text = gr.Textbox(
                    lines=3, 
                    placeholder="Suggestions for improvement (optional)",
                    label="Feedback"
                )
                feedback_btn = gr.Button("Submit Feedback")
                feedback_output = gr.Textbox(label="")
        
        example_dreams = [
            ["I was flying over a vast ocean, feeling both exhilarated and afraid. Suddenly I started falling, but before hitting the water I woke up."],
            ["I was in my childhood home, but the rooms were bigger and had doors I had never seen before. I opened one door and found a room full of old toys."],
            ["I was being chased through a dark forest by a figure I couldn't see clearly. I kept running but my legs felt heavy and I couldn't move fast enough."],
            ["I was taking an important exam but realized I hadn't studied at all. The questions made no sense and the clock was ticking loudly. Everyone else seemed to know the answers."],
            ["I was at a party with friends when I suddenly realized I wasn't wearing any clothes. Everyone was talking normally as if nothing was wrong, but I felt deeply embarrassed."]
        ]
        
        gr.Examples(
            examples=example_dreams,
            inputs=dream_input
        )
        
        analyze_btn.click(
            fn=analyze_dream_fn,
            inputs=dream_input,
            outputs=analysis_output
        )
        
        feedback_btn.click(
            fn=submit_feedback,
            inputs=[dream_input, analysis_output, feedback_rating, feedback_text],
            outputs=feedback_output
        )
        
        # Custom CSS for better formatting
        interface.load(None, None, None, _js="""
        () => {
            // Add custom CSS for better styling
            const style = document.createElement('style');
            style.textContent = `
                h3 {
                    color: #4b6cb7;
                    margin-top: 1.5em;
                    margin-bottom: 0.7em;
                }
                ul {
                    margin-left: 1.5em;
                }
                li {
                    margin-bottom: 0.5em;
                }
                #analysis_output {
                    line-height: 1.6;
                    font-size: 1.05em;
                }
            `;
            document.head.appendChild(style);
        }
        """)
        
    return interface