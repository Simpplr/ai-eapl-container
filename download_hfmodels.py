from sentence_transformers import SentenceTransformer

models = [
    'sentence-transformers/roberta-base-nli-stsb-mean-tokens',
    'sentence-transformers/all-MiniLM-L6-v2'
]

for model in models:
    SentenceTransformer(model)

