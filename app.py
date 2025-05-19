import os
import sys
import argparse
import gradio as gr

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from ui.dream_interface import create_dream_interface

def main():
    parser = argparse.ArgumentParser(description="Freudian Dream Analyzer Interface")
    parser.add_argument("--model_path", type=str,
                        default="model/unsloth_model/final_model",
                        help="Path to the fine-tuned model")
    parser.add_argument("--hf_space", action="store_true",
                        help="Running in Hugging Face Space")
    args = parser.parse_args()
    
    if args.hf_space:
        if os.path.exists("/model"):
            model_path = "/model"
        else:
            model_path = "model"
        print(f"Running in Hugging Face Spaces, using model path: {model_path}")
    else:
        model_path = os.path.join(current_dir, "model", "unsloth_model", "final_model")
        print(f"Running locally, using model path: {model_path}")
    
    demo = create_dream_interface(model_path)
    demo.launch()

if __name__ == "__main__":
    main()