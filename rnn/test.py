"""
test.py

Feed a dataset of sequences to the RNN and check for correct predictions
"""

from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from misc import (check_accuracy, check_sequence_correctness, sample_from_rnn,
                  translate_tokens)
from model import DigitSumModel
from omegaconf import DictConfig
from torch import nn

from data import EOS_IDX, VOCAB_SIZE, DigitSequenceDataset


def test_model_on_sequence(model: nn.Module, test_seuqence: str) -> str:
    """ Convert the test_sequence into a tensor, pass it through the model and get the answer """

    # convert the test_sequence to a tensor
    seq_list = list(map(int, list(test_seuqence)))
    seq_list.append(EOS_IDX)
    seq_tensor = F.one_hot(torch.tensor(seq_list), num_classes=VOCAB_SIZE).float()

    answer_seq = sample_from_rnn(model, seq_tensor)
    answer_str = translate_tokens(answer_seq)
    answer_correct = check_sequence_correctness(seq_tensor, answer_str)
    return answer_str, answer_correct


@hydra.main(version_base=None, config_name="config", config_path=".")
def main(cfg: DictConfig) -> None:
    """ main function, check configs, load model and dataset """

    # check for model type
    if not cfg.model.model_type or not isinstance(cfg.model.model_type, str):
        print("Please provide valid model type with the test.model_type argument!"
              " Options: [rnn, lstm]")
        return
    model_type = cfg.model.model_type
    
    # check if hidden size is given, is an integer and not too big or too small
    if not cfg.model.hidden_size:
        print("No train.hidden_size argument given, using default value of 128")
        hidden_size = 128

    if not isinstance(cfg.model.hidden_size, int) or not 16 <= cfg.model.hidden_size <= 256:
        print("train.hidden_size argument must be an integer between 16 and 256!")
        return
    hidden_size = cfg.model.hidden_size

    # check if both dataset_path and string are provided
    if cfg.test.dataset_path is not None and cfg.test.sequence is not None:
        print("Please provide only one of 'test.dataset_path' or 'test.sequence'")
        return

    # check if neither dataset path nor sequence is provided
    if not cfg.test.dataset_path and not cfg.test.sequence:
        print("Please provide one of 'test.dataset_path' or 'test.sequence' arguments")
        return

    # if only a test sequence is provided
    if cfg.test.dataset_path is not None:
        # check if dataset path exists
        if not Path(cfg.test.dataset_path).exists():
            print(f"Dataset at {cfg.test.dataset_path} does not exist")
            return

        # load dataset
        ds = DigitSequenceDataset(cfg.test.dataset_path)
        test_sequence = None
    else:
        test_sequence = str(cfg.test.sequence)
        ds = None

    # check if model path is provided
    if not cfg.test.model_path:
        print("Please provide path to model with the 'test.model_path' argument")
        return

    # check if model path exists
    if not Path(cfg.test.model_path).exists():
        print(f"Model at {cfg.test.model_path} does not exist")
        return
    # load model
    model = DigitSumModel(VOCAB_SIZE, hidden_size, VOCAB_SIZE, model_type=model_type)
    try:
        model.load_state_dict(torch.load(cfg.test.model_path, weights_only=True))
    except Exception as e:
        print(f"Could not load model at path {cfg.test.model_path}.")
        print(f"Error: {e}")

    # if a test_sequence is provided, run the model on it
    if test_sequence is not None:
        print()
        print("Test model on input:")
        print(test_sequence)
        answer, correct = test_model_on_sequence(model, test_sequence)
        print(f"Answer: {answer}, {correct}")
    else:
        # check accuracy of the model on dataset
        acc = check_accuracy(model, ds, n_info=50)
        print()
        print(f"Accuracy on dataset: {cfg.test.dataset_path}")
        print(f"{acc*100:.2f}%")


if __name__ == "__main__":
    main()
