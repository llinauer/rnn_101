""" Create a dataset of all sequences of digits with length 2 to k as inputs and
    their sum as labels
    E.g. x1=00 y1=0
         x2=01 y2=1
         ...
         xN=999..99 (k digits) yN=9*k
    Save the data in a csv file
"""

import sys
import csv


def create_sequences_of_length_k(k: int) -> list:
    """ Create all sequences of k digits and return them as a list """
    sequences = []
    for i in range(0, 10**k):
        sequences.append(str(i).zfill(k))
    return sequences


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
    if len(sys.argv) != 2:
        print("Usage: python create_dataset.py <seq_length>")
        sys.exit()

    # get number of samples and sample length from args
    seq_len = int(sys.argv[1])

    # restrict k to maximum of 8
    if seq_len >= 8 or seq_len < 2:
        print("<seq_length> must be > 2 and <= 8")
        sys.exit()

    print(f"Creating dataset of sequences with {seq_len} digits")
    data = []
    for k in range(2, seq_len+1):
        sequences = create_sequences_of_length_k(k)
        sums = list(map(calc_digit_sum, sequences))
        data.extend(zip(sequences, sums))

    # save data to csv
    with open("seq_data.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print("Saved data to 'seq_data.csv'")


if __name__ == "__main__":
    main()
