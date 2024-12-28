"""
test.py

Feed a dataset of sequences to the RNN and check for correct predictions
"""

from pathlib import Path

import hydra
import torch
from data import VOCAB_SIZE, DigitSequenceDataset
from misc import check_sequence_correctness, sample_from_rnn, translate_tokens
from model import DigitSumModel
from omegaconf import DictConfig
from torch import nn


def check_accuracy(model: nn.Module, dataset: DigitSequenceDataset, n_info: int = 50) -> float:
    """ Loop over the dataset, feed the input sequences to the model and check if it produces
        the correct output
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


@hydra.main(version_base=None, config_name="config", config_path=".")
def main(cfg: DictConfig) -> None:
    """ main function, check configs, load model and dataset """

    # check if dataset path is provided
    if not cfg.test.dataset_path:
        print("Please provide path to dataset with the 'test.dataset_path' argument")
        return

    # check if dataset path exists
    if not Path(cfg.test.dataset_path).exists():
        print("Dataset at {cfg.test.dataset_path} does not exist")
        return

    # load dataset
    ds = DigitSequenceDataset(cfg.test.dataset_path)

    # check if model path is provided
    if not cfg.test.model_path:
        print("Please provide path to model with the 'test.model_path' argument")
        return

    # check if model path exists
    if not Path(cfg.test.model_path).exists():
        print("Dataset at {cfg.test.model_path} does not exist")
        return

    # load model
    model = DigitSumModel(VOCAB_SIZE, 128, VOCAB_SIZE)
    try:
        model.load_state_dict(torch.load(cfg.test.model_path, weights_only=True))
    except:
        print("Could not load model at path {cfg.test.model_path}")

    # check accuracy of the model on dataset
    acc = check_accuracy(model, ds)
    print()
    print(f"Accuracy on dataset: {cfg.test.dataset_path}")
    print(f"{acc*100:.2f}%")


if __name__ == "__main__":
    main()
