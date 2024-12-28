"""
model.py

RNN model definition
"""

from torch import nn


class DigitSumModel(nn.Module):
    """ RNN that takes a sequence of digits as an input and predicts their sum """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        # define layers
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_0=None):
        hidden, _ = self.rnn(x, h_0)
        logits = self.h2o(hidden)

        return logits, hidden
