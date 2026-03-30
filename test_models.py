import torch
from models.load_models import load_student, get_hidden_states

def main():
    # Load the student model (Qwen/Qwen2.5-1.5B)
    model, tokenizer = load_student("Qwen/Qwen2.5-0.5B")
    
    # Use the device the auto device_map picked
    device = next(model.parameters()).device
    
    hidden_states, last_hidden, input_ids = get_hidden_states(
        model, tokenizer, "What is 2 + 2?", device=device
    )
    
    print("Hidden states shape:", last_hidden.shape)
    print("Num layers:", len(hidden_states))
    print("Model test passed ✅")

if __name__ == "__main__":
    main()
