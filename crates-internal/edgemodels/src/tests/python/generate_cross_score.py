import argparse
from sentence_transformers.cross_encoder import CrossEncoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--document", type=str, required=True)
    args = parser.parse_args()
    
    model = CrossEncoder(args.model, device='cpu')
    score = model.predict([(args.query, args.document)])
    print(float(score[0]))

if __name__ == "__main__":
    main()
    
    