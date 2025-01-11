"""
misc.py

Miscellaneous module for functions used by both train.py and test.py
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from data import EOA_IDX, EOS_IDX, VOCAB_SIZE, DigitSequenceDataset


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


def check_accuracy(model: nn.Module, dataset: DigitSequenceDataset,
                   n_info: Optional[int] = None) -> float:
    """ Loop over the dataset, feed the input sequences to the model and check if it produces
        the correct output. Every n_info iterations, print a sequence and it's predictions,
        unless n_info = None
    """

    correct_answers = 0.

    # loop over whole dataset
    for i, (input_seq, answer_seq) in enumerate(dataset):
        # feed input sequence into model
        answer_seq = sample_from_rnn(model, input_seq)
        # translate answer tokens to string
        answer_str = translate_tokens(answer_seq)
        # check if answer is correct
        answer_correct = check_sequence_correctness(input_seq, answer_str)
        correct_answers += float(answer_correct)

        # if n_info==None, do not print
        if not n_info:
            continue

        # for a feeling of progress, print answers every n iterations
        if i % n_info == 0:
            # print the input sequence and the generated tokens
            input_seq_str = translate_tokens(input_seq)
            print(f"Step {i}/{len(dataset)}")
            print("Input sequence: ", input_seq_str)
            print("Answer: ", answer_str)
            print(answer_correct)

    accuracy = correct_answers / len(dataset)
    return accuracy
