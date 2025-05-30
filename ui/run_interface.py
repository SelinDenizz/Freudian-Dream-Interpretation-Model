import os
import sys
import argparse
import gradio as gr

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from ui.dream_interface import create_dream_interface

def main():
    parser = argparse.ArgumentParser(description="Freudian Dream Analyzer Interface")
    parser.add_argument("--model_path", type=str,
                        default="model/unsloth_model/final_model",
                        help="Path to the fine-tuned model (relative to project root)")
    parser.add_argument("--share", action="store_true",
                        help="Create a shareable link")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode")
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
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, args.model_path)
        print(f"Running locally, using model path: {model_path}")
    
    interface = create_dream_interface(model_path)
    
    if args.debug:
        interface.launch(server_name="0.0.0.0", server_port=7860, share=args.share, debug=True)
    else:
        interface.launch(share=args.share)

if __name__ == "__main__":
    main()