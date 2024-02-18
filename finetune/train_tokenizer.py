import sentencepiece as spm

spm.SentencePieceTrainer.Train('--input=./wiki.txt --model_prefix=wiki_tokenizer --vocab_size=10000 --model_type=bpe')