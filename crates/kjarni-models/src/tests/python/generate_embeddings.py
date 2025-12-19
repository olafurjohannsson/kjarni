import argparse
import json
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--sentences", type=str, nargs='+', required=True)
    args = parser.parse_args()

    model = SentenceTransformer(args.model, device='cpu')
    embeddings = model.encode(args.sentences, convert_to_tensor=False)
    print(json.dumps(embeddings.tolist()))

if __name__ == "__main__":
    main()