#!/usr/bin/env python3
"""
Inference script for GPRO Empathy model.
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpro_empathy.utils.inference import EmpathyInference


def main():
    parser = argparse.ArgumentParser(description="Run inference with GPRO Empathy model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="grpo_saved_lora",
        help="Path to saved LoRA adapter"
    )
    parser.add_argument(
        "--message",
        type=str,
        help="User message to respond to"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    print("=== Loading GPRO Empathy Model ===")
    print(f"Config: {args.config}")
    print(f"LoRA: {args.lora_path}")
    
    # Initialize inference model
    inference = EmpathyInference(
        config_path=args.config,
        lora_path=args.lora_path
    )
    
    if args.interactive:
        print("\n=== Interactive Mode ===")
        print("Type your messages and get empathetic responses.")
        print("Type 'quit' or 'exit' to end the session.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                print("Assistant:", end=" ")
                response = inference.generate_empathetic_response(
                    user_input,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                )
                
                # Extract just the answer part if XML formatting is used
                clean_response = inference.extract_answer_from_response(response)
                print(clean_response)
                print()
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    elif args.message:
        print(f"\n=== Single Message Response ===")
        print(f"User: {args.message}")
        
        response = inference.generate_empathetic_response(
            args.message,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        
        # Extract just the answer part if XML formatting is used
        clean_response = inference.extract_answer_from_response(response)
        print(f"Assistant: {clean_response}")
        
        # Also show full response for debugging
        print(f"\nFull response:\n{response}")
    
    else:
        print("Please specify either --message for single response or --interactive for chat mode")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())