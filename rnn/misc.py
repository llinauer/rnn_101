"""
misc.py

Miscellaneous module for functions used by both train.py and test.py
"""

import torch
from torch import nn
import torch.nn.functional as F
from data import EOA_IDX, EOS_IDX, VOCAB_SIZE


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
    hidden_states = hidden_states

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


def translate_tokens(tokens: torch.Tensor) -> str:
    """ Translate the tokens back to the vocab and print
        tokens is of shape (n_tokens, d_vocab) """

    # convert sequence of tensor to list of integers
    digit_list = tokens.argmax(dim=1).tolist()

    # join the list to a string
    digit_string = " ".join(map(str, digit_list))

    # replace 10 with EOA and 11 with EOS
    digit_string = digit_string.replace(str(EOA_IDX), "EOA")
    digit_string = digit_string.replace(str(EOS_IDX), "EOS")
    return digit_string


def check_sequence_correctness(input_sequence: torch.Tensor, answer: str) -> bool:
    """ Check if the generated answer for the input_sequence is correct
        The shape of input_sequence is (n_sequence, d_vocab) """

    # calc sum of input_sequence tokens, ignore the EOS token
    digit_sum = input_sequence.argmax(dim=1)[:-1].sum().item()
    answer = answer.replace(" ", "").replace("EOA", "").replace("EOS", "")
    # if the answer is just EOA or EOS, set to 0
    if not answer:
        answer = 0
    else:
        answer = int(answer)
    return digit_sum == answer
