import argparse
import json
from sentence_transformers.cross_encoder import CrossEncoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--documents", type=str, nargs='+', required=True)
    args = parser.parse_args()

    model = CrossEncoder(args.model, device='cpu')
    sentence_combinations = [[args.query, doc] for doc in args.documents]
    scores = model.predict(sentence_combinations)
    ranked_results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    ranked_indices = [item[0] for item in ranked_results]
    print(json.dumps(ranked_indices))

if __name__ == "__main__":
    main()