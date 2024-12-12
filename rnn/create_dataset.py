""" Create a dataset of N sequences of digits with length of k as inputs and
    their sum as labels
    E.g. x1=12345 y1=15
         x2=44444 y2=20
         ...
    Save the data in a csv file
"""

import sys
import numpy as np


def calc_digit_sum(digit_str: str) -> str:
    """ Calculate the digit sum of the number digit_str, represented as a string
        e.g. '1234' -> 10
    """
    digit_list = list(digit_str)
    # map each entry of digit list to an integer and sum them
    digit_sum = sum(map(int, digit_list))

    # return the sum as a string
    return str(digit_sum)


def main():
    """ Main function, create data """

    # check if args supplied
    if len(sys.argv) != 3:
        print("Usage: python create_dataset.py <num_samples> <seq_length>")
        sys.exit()

    # get number of samples and sample length from args
    n_samples, seq_len = sys.argv[1:]

    # restrict k to maximum of 20
    if int(seq_len) >= 20:
        print("<seq_length> must be <= 20")
        sys.exit()

    print(f"Creating {int(n_samples)} samples of sequences with {seq_len} digits")

    # create n_samples numbers of pre-determined length
    seqs = np.random.randint(low=10**(int(seq_len)-1) + 1, high=10**int(seq_len),
                             size=int(n_samples))

    # seqs is now an array containing integers
    # to get the labels, calculate the sum of the digits of each sample
    seqs_string = seqs.astype(str)
    calc_digit_sum_vec = np.vectorize(calc_digit_sum)
    labels = calc_digit_sum_vec(seqs_string)

    # save the sequences and the sums in a csv file
    combined_array = np.column_stack((seqs_string, labels))
    np.savetxt("seq_data.csv", combined_array, delimiter=",", fmt="%s")

    print("Saved data to 'seq_data.csv'")


if __name__ == "__main__":
    main()
