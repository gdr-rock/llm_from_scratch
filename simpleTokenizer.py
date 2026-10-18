import re

# Simple Text Tokenizer
class SimpleTokenizer:
    def __init__(self, vocab=None):
        if vocab is None:
            self.vocab = {"<UNK>": 0}
        else:
            self.vocab = vocab

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        tokens = re.split(r'([,.:?_!"()\']|--|\s)', text)

        return [
            token
            for token in tokens
            if token.strip() != ""
        ]

    def detokenize(self, tokens):
        text = " ".join(tokens)

        text = re.sub(
            r'\s([,.:?_!"()\'])',
            r'\1',
            text
        )

        return text

    def encode(self, text):
        tokens = self.tokenize(text)

        token_ids = [
            self.vocab.get(
                token,
                self.vocab["<UNK>"]
            )
            for token in tokens
        ]

        return token_ids

    def decode(self, token_ids):
        tokens = [
            self.reverse_vocab[token_id]
            for token_id in token_ids
        ]

        return self.detokenize(tokens)


text = "Hello, world! This is a test."

tokenizer = SimpleTokenizer()
# text -> tokens
tokens = tokenizer.tokenize(text)
print("Tokens:")
print(tokens)

# tokens -> text
detokenized = tokenizer.detokenize(tokens)
print("\nDetokenized:")
print(detokenized)

# text -> IDs
token_ids = tokenizer.encode(text)
print("\nToken IDs:")
print(token_ids)

# IDs -> text
decoded = tokenizer.decode(token_ids)
print("\nDecoded:")
print(decoded)