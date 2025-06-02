import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from ui.dream_interface import create_dream_interface

def main():
    logger.info("Starting Freudian Dream Analyzer with ZeroGPU support")
    
    model_path = "model/unsloth_model"
    
    logger.info(f"Using model path: {model_path}")
    if os.path.exists(model_path):
        logger.info("Model directory exists")
        try:
            for item in sorted(os.listdir(model_path)):
                logger.info(f"  - {item}")
        except Exception as e:
            logger.error(f"Could not list model directory: {e}")
    else:
        logger.warning("Model directory does not exist!")
    
    logger.info("Creating Gradio interface...")
    try:
        interface = create_dream_interface(model_path)
        logger.info("Gradio interface created successfully")
    except Exception as e:
        logger.error(f"Failed to create Gradio interface: {e}")
        raise
    
    logger.info("Launching interface...")
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=True
    )

if __name__ == "__main__":
    main()