import sentencepiece as spm

def load_tokenizer(vocab_file="wiki_tokenizer.model"):
    sp = spm.SentencePieceProcessor()
    sp.load(vocab_file)

    return sp

def tokenize(tokenizer, prompt):
    return tokenizer.encode_as_ids(prompt)