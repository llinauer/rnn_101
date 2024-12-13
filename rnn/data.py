"""
data.py

Create a custom torch Dataset for the digit sequence dataset
"""

from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


VOCAB_SIZE = 12
EOS_IDX = 10
EOA_IDX = 11


class DigitSequenceDataset(Dataset):
    """ DigitSequenceDataset class
        Reads in data from a csv, yields tensors of shape (seq_len, vocab_len) as inputs
        and tensors of shape (seq_len, vocab_len) as labels

        The vocab consists of 12 elements: the digits from 0-9, an <EOS> and an <EOA> token
    """

    def __init__(self, csv_file_path: str) -> None:
        self.data_df = pd.read_csv(csv_file_path, names=["x", "y"], dtype=str)
        self.vocab_size = VOCAB_SIZE

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx) -> Any:

        # get seq and label with index idx from dataframe
        seq = self.data_df.iloc[idx]["x"]
        label = self.data_df.iloc[idx]["y"]

        # convert them to a list of digits
        seq_list = list(map(int, list(seq)))
        label_list = list(map(int, list(label)))

        # add a <EOS> token to the end of the seq list to indicate, that the sequence is over
        seq_list.append(EOS_IDX)

        # add an <EOA> token to the end of the label, to indicate that the answer is over
        label_list.append(EOA_IDX)

        # create tensors out of seq and label lists and one-hot encode them
        seq_tensor = F.one_hot(torch.tensor(seq_list), num_classes=self.vocab_size)
        label_tensor = F.one_hot(torch.tensor(label_list), num_classes=self.vocab_size)

        return seq_tensor.float(), label_tensor.float()
