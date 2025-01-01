"""
data.py

Create a custom torch Dataset for the digit sequence dataset
"""

from typing import Any, List, Tuple, Iterator

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from random import shuffle


VOCAB_SIZE = 12
EOS_IDX = 10
EOA_IDX = 11


def get_lengths(row: pd.Series) -> Tuple[int, int]:
    """ Get the length of the input sequence and the target sequence of a row in the dataframe """
    return (len(row["x"]), len(row["y"]))


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


class EqualSequenceLengthSampler(Sampler):
    """ Custom sampler class, to batch sequences of equal length and targets of equal length """

    def __init__(self, data: Dataset) -> None:
        """ Constructor """

        # store the whole dataframe to get the total number of samples from __len__
        self.df = data.data_df.copy()

        # group the elements of the dataframe according to the length of the input as well as
        # target sequence
        self.df["lengths"] = self.df.apply(get_lengths, axis=1)
        grouped_df = self.df.groupby("lengths")

        # grouped_df is now grouped based on both the length of the input and the length of the
        # target sequence
        # e.g. if we have two samples (12345, 15) & (666, 18), the first belongs to the group
        # (5, 2) and the second to the group (3, 2)
        # this ensures that we can always batch together all inputs and targets of one group

        # create a list of lists, where the inner lists contain the indices of the individual groups
        self.idx_list = [group.index.tolist() for _, group in grouped_df]

    def shuffle_indices(self) -> None:
        """ Shuffle the order of groups in self.sidx_list and the order of the indices within
            the groups """
        for group in self.idx_list:
            shuffle(group)
        shuffle(self.idx_list)

    def __len__(self) -> int:
        """ Get the total number of elements in the dataframe -> needed for calling len on 
            the dataloader """
        return len(self.df)

    def __iter__(self) -> Iterator[List[int]]:
        # shuffle the whole data before creating an iterator
        self.shuffle_indices()
        return iter(self.idx_list)
