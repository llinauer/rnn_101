"""
model.py

RNN model definition
"""

from torch import nn


class DigitSumModel(nn.Module):
    """ RNN that takes a sequence of digits as an input and predicts their sum.
        Uses either fully-connected RNN or LSTM"""

    def __init__(self, input_size, hidden_size, output_size, model_type="rnn"):
        super().__init__()

        # define layers
        if model_type == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif model_type == "rnn":
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError("Invalid model_type")
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_0=None):
        output, last_hidden = self.rnn(x, h_0)
        logits = self.h2o(output)

        return logits, last_hidden
