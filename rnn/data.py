"""
data.py

Create a custom torch Dataset for the digit sequence dataset
"""

from typing import Any, List, Tuple, Iterator

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, Sampler, BatchSampler
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


class EqualLengthSampler(Sampler):
    """ Custom sampler class, to batch sequences of equal length and targets of equal length """

    def __init__(self, data: Subset, shuffle=False) -> None:
        """ Constructor. Takes in a pytorch.utils.data.Subset which corresponds to
            either train or validation split.
            Can choose to not shuffle the data
        """

        # store the subset of the whole dataframe, according to the train or validation split
        self.df = data.dataset.data_df.copy().iloc[data.indices]
        # reset indices, so that they are in the range [0, len(subset)]
        self.df = self.df.reset_index()

        # store other attributes
        self.shuffle = shuffle

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
        self.batch_idx_list = [group.index.tolist() for _, group in grouped_df]

        # map indices to groups
        self.idx_to_group = {}
        for group, idx_list in enumerate(self.batch_idx_list):
            for idx in idx_list:
                self.idx_to_group[idx] = group

    def shuffle_indices(self) -> None:
        """ Shuffle the order of groups in self.batch_idx_list and the order of the indices within
            the groups """

        # check if should shuffle
        if not self.shuffle:
            return

        for group in self.batch_idx_list:
            shuffle(group)
        shuffle(self.batch_idx_list)

    def __len__(self) -> int:
        """ Get the number of groups """
        return len(self.batch_idx_list)

    def __iter__(self) -> Iterator[List[int]]:
        # shuffle the whole data before creating an iterator
        self.shuffle_indices()
        for idx_list in self.batch_idx_list:
            for idx in idx_list:
                yield idx


class EqualLengthBatchSampler(BatchSampler):
    """ Custom Batch Sampler, to pack input sequences of equal length and target sequences of
         equal length together
    """

    def __init__(self, sampler: Sampler, batch_size: int):
        """ Constructor """

        self.sampler = sampler
        self.batch_size = batch_size

    def __iter__(self):
        """ Create iterator. Iterate over the indices in the sampler attribute. Check
             for indices belonging to the same group and yield batches of them .
        """
        batch = []
        current_group = None

        for idx in self.sampler:
            group = self.sampler.idx_to_group[idx]
            if current_group is None:
                current_group = group

            if group != current_group or len(batch) == self.batch_size:
                yield batch
                batch = []
                current_group = group

            batch.append(idx)

        # yield any remaining indices
        if batch:
            yield batch

    def __len__(self):
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.sampler.batch_idx_list
        )
