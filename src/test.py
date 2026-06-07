from sentence_transformers import SentenceTransformer

from corpus_loader import load_corpus

model = SentenceTransformer("all-MiniLM-L6-v2")
print(model.max_seq_length)  # Should print 256
print(model.tokenizer("Here is a test sentence of a few words")["input_ids"])


docs = load_corpus("../corpus")
print(len(docs))
all_text = " ".join(d.content for d in docs)
words = all_text.split()
tokens = model.tokenizer(all_text, truncation=False)["input_ids"]
print(f"Ratio: {len(tokens) / len(words):.2f} tokens per word")
