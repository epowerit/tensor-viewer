"""Word embeddings.

Each token id picks one row of a learned table. Averaging the rows gives one
vector for the whole sentence, which a linear layer classifies.
"""

from torch import nn

# track: Foundations
# input tokens: text "the cat sat on the mat and the dog slept on the rug"


class BagOfWords(nn.Module):
    def __init__(self, vocabulary=64, dim=8, classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, dim)
        self.classifier = nn.Linear(dim, classes)

    def forward(self, tokens):
        vectors = self.embedding(tokens)  # axes: batch, tokens, features
        sentence = vectors.mean(dim=1)  # axes: batch, features
        return self.classifier(sentence).softmax(dim=-1)
