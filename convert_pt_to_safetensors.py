
import torch
import safetensors.torch
import sys
import os

def convert_pt_to_safetensors(pt_path, sf_path):
    print(f"Loading {pt_path}...")
    try:
        # Load the PyTorch checkpoint
        checkpoint = torch.load(pt_path, map_location="cpu")
        
        # Extract state_dict
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Clean up keys if necessary (e.g., remove "module." prefix)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[new_key] = v
            
        print(f"Saving to {sf_path}...")
        safetensors.torch.save_file(new_state_dict, sf_path)
        print("Conversion complete.")
        
    except Exception as e:
        print(f"Error converting file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_pt_to_safetensors.py <input.pt> [output.safetensors]")
        sys.exit(1)
        
    input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = os.path.splitext(input_path)[0] + ".safetensors"
        
    convert_pt_to_safetensors(input_path, output_path)
