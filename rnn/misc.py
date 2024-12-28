"""
misc.py

Miscellaneous module for functions used by both train.py and test.py
"""

import torch
from torch import nn
import torch.nn.functional as F
from data import EOA_IDX, VOCAB_SIZE


def sample_from_rnn(model: nn.Module, input_sequence: torch.Tensor,
                    max_seq_len: int = 5) -> torch.Tensor:
    """ Sample from an RNN, by giving it an input_sequence and iteratively generate new tokens
        input_sequence is of shape (n_sequence, d_vocab) """

    # pass input_sequence through model
    with torch.no_grad():
        next_logits, hidden_states = model(input_sequence)
    # process the logits of the last input, to get the next input token
    next_token = F.one_hot(next_logits[-1].argmax(), num_classes=VOCAB_SIZE).float()
    generated_tokens = [next_token]
    hidden_states = hidden_states[-1:]

    if torch.equal(next_token, F.one_hot(torch.tensor(EOA_IDX)).float()):
        return torch.vstack(generated_tokens)

    # loop until the max number of generated tokens is reached or we encounter an EOA token
    for _ in range(max_seq_len - 1):

        # put tokens in rnn
        next_logits, hidden_states = model(next_token.unsqueeze(0), hidden_states)
        next_token = F.one_hot(next_logits[-1].argmax(), num_classes=VOCAB_SIZE).float()

        generated_tokens.append(next_token)

        # check if EOA token was generated
        if torch.equal(next_token, F.one_hot(torch.tensor(EOA_IDX)).float()):
            break

    return torch.vstack(generated_tokens)
