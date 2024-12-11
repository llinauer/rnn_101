"""
data.py

Create a custom torch Dataset for the digit sequence dataset
"""

from typing import Any
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F


class DigitSequenceDataset(Dataset):
    """ DigitSequenceDataset class
        Reads in data from a csv, yields tensors of shape (seq_len, vocab_len) as inputs
        and tensors of shape (seq_len, vocab_len) as labels

        The vocab consists of 11 elements: the digits from 0-9 and an <EOS> token
    """

    def __init__(self, csv_file_path: str) -> None:
        self.data_df = pd.read_csv(csv_file_path, names=["x", "y"], dtype=str)
        self.vocab_size = 11

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx) -> Any:

        # get seq and label with index idx from dataframe
        seq = self.data_df.iloc[idx]["x"]
        label = self.data_df.iloc[idx]["y"]

        # convert them to a list of digits
        seq_list = list(map(int, list(seq)))
        label_list = list(map(int, list(label)))

        seq_tensor = F.one_hot(seq_list, num_classes=self.vocab_size)
        label_tensor = F.one_hot(label_list, num_classes=self.vocab_size)

        # create tensors out of seq and label lists and one-hot encode them
        return seq_tensor, label_tensor
