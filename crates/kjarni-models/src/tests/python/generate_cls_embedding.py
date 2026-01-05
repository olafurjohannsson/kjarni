import argparse
import json
from sentence_transformers import SentenceTransformer
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--sentence", type=str, required=True)
    args = parser.parse_args()

    model = SentenceTransformer(args.model, device='cpu')
    
    encoded_input = model.tokenizer(args.sentence, padding=True, truncation=True, return_tensors='pt')
    
    # The first module is the transformer model.
    transformer_model = model[0]
    
    with torch.no_grad():
        # This call is correct.
        model_output = transformer_model(encoded_input)
    
    # --- THIS IS THE FIX ---
    # The dictionary key for the token embeddings in the sentence-transformers
    # library is 'token_embeddings', not 'last_hidden_state'.
    last_hidden_state = model_output['token_embeddings']
    # --- END FIX ---
    
    # This logic is now correct because `last_hidden_state` is the correct tensor.
    cls_embedding = last_hidden_state[:,0,:].squeeze().tolist()
    
    print(json.dumps(cls_embedding))

if __name__ == "__main__":
    main()