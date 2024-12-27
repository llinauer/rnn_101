"""
train.py

Train an RNN to predict the digit sum of a sequence
"""

import random
from pathlib import Path
from time import gmtime, strftime

import einops
import hydra
import torch
import torch.nn.functional as F
from data import EOA_IDX, EOS_IDX, VOCAB_SIZE, DigitSequenceDataset
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class OneHotCrossEntropyLoss(nn.Module):
    """ Custom class for calculating Cross entropy with one-hot encoded targets """
    def forward(self, logits, one_hot_targets):
        log_probs = F.log_softmax(logits, dim=-1)
        cross_entropy = -einops.einsum(log_probs, one_hot_targets,
                                       "batch sequence vocab, batch sequence vocab -> batch sequence")
        loss = cross_entropy.mean()
        return loss


class DigitSumModel(nn.Module):
    """ RNN that takes a sequence of digits as an input and predicts their sum """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        # define layers
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_0=None):
        hidden, _ = self.rnn(x, h_0)
        logits = self.h2o(hidden)

        return logits, hidden


def sample_from_rnn(model, input_sequence, max_seq_len=5):
    """ Sample from an RNN, by giving it an input_sequence and iteratively generate new tokens
        input_sequence is of shape (n_sequence, d_vocab) """

    # pass input_sequence through model
    with torch.no_grad():
        next_logits, hidden_states = model(input_sequence)
    # process the logits of the last input, to get the next input token
    next_token = F.one_hot(next_logits[-1].argmax(), num_classes=VOCAB_SIZE).float()
    generated_tokens = [next_token]
    hidden_states = hidden_states[-1:]

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


def train(model, train_loader, val_loader, loss_func, optimizer, n_epochs, log_path, tb_logger):
    """ Train function. Iterate over the batch_loader epochs times and train the model.
        Log metrics with tensorboard logger """

    # loop for n_epochs
    for epoch in range(n_epochs):

        train_loss = 0.0
        val_loss = 0.0
        best_model_performance = -torch.inf

        # train loop
        # move model to appropriate device and set to train mode
        model.train()

        # create progress bar out of train_loader
        train_progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                                  desc=f"Epoch {epoch+1}/{n_epochs}")

        for _, (input_sequences, target_sequences) in train_progress_bar:

            # reset gradients
            optimizer.zero_grad()

            # pass input_sequences through model to get the first prediction and the hidden state
            input_logits, input_hidden_states = model(input_sequences)

            # pass the target sequences through the model to get further predictions
            target_logits, _ = model(
                target_sequences, h_0=input_hidden_states[:, -1, :].unsqueeze(0)
                )

            # calculate the loss
            # first, take the prediction of the RNN at the end of the input_sequence 
            # -> should be equal to the first element of the target_sequence
            loss = loss_func(input_logits[:, -2:-1, :], target_sequences[:, 0:1, :])

            # for each prediction in the target sequence, compare with the following elements
            # of the target sequence
            loss += loss_func(target_logits[:, :-1, :], target_sequences[:, 1:, :])

            # backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # accumulate training loss
            train_loss += loss.item()

            # print running training loss
            train_progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # log train loss
        tb_logger.add_scalar('Loss/train', avg_train_loss, epoch)

        # every 10 epochs, inform the user
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Average Loss: {avg_train_loss:.4f}")

        # validation loop, set model to eval mode
        model.eval()

        # create progress bar out of val_loader
        val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))
        for _, (input_sequences, target_sequences) in val_progress_bar:

            # pass input_sequences and target_sequences through the model
            input_logits, input_hidden_states = model(input_sequences)
            # pass the target sequences through the model to get further predictions
            target_logits, _ = model(
                target_sequences, h_0=input_hidden_states[:, -1, :].unsqueeze(0)
                )

            # calculate loss like in training loop
            loss = loss_func(input_logits[:, -2:-1, :], target_sequences[:, 0:1, :])
            loss += loss_func(target_logits[:, :-1, :], target_sequences[:, 1:, :])
            val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        # log val loss
        tb_logger.add_scalar("Loss/val", avg_val_loss, epoch)

        # check if model performance improved
        model_performance = -avg_val_loss
        if model_performance > best_model_performance:
            print(f"New best model performance. Saving model to {log_path/'best_model.pth'}")
            torch.save(model.cpu().state_dict(), log_path / "best_model.pth")

        # every n epochs, sample from the RNN and check if the calculation is correct
        if epoch % 10 == 0:
            print(f"Validation: Average Loss: {val_loss/len(val_loader):.4f}")

            # check current model output on a randomly sampled sequence from the validation set
            random_batch_idx = random.randint(0, len(val_loader)-1)
            random_batch = list(val_loader)[random_batch_idx][0]
            random_sample_idx = random.randint(0, len(random_batch)-1)
            random_sample = random_batch[random_sample_idx, :, :]
            generated_tokens = sample_from_rnn(model, random_sample)

            # print the sequence and the generated tokens
            input_seq_str = translate_tokens(random_sample)
            print("Input sequence: ", input_seq_str)
            answer_str = translate_tokens(generated_tokens)
            print("Answer: ", answer_str)
            answer_correct = check_sequence_correctness(random_sample, answer_str)
            print(answer_correct)

            # log the RNN calculation to tensorboard
            log_str = f"Input Sequence: {input_seq_str}, Answer: {answer_str} ->  {answer_correct}"
            tb_logger.add_text("RNN output", log_str, epoch)


@hydra.main(version_base=None, config_name="config", config_path=".")
def main(cfg: DictConfig) -> None:
    """ Main function """

    log_path = Path(cfg.log_path)

    # check if run name was given, if not create from current time
    if not cfg.run_name:
        run_name = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    else:
        run_name = cfg.run_name

    # create a path for each run
    log_path = log_path / run_name

    # create log_path if it does not exist
    if not log_path.exists():
        log_path.mkdir(parents=True, exist_ok=True)

    # create tensorboard logger
    logger = SummaryWriter(log_path)

    # check if dataset path is provided
    if not cfg.dataset_path:
        print("Please provide path to dataset with the 'dataset_path' argument")
        return

    # check if dataset path exists
    if not Path(cfg.dataset_path).exists():
        print("Dataset at {ds_path} does not exist")
        return

    # load dataset
    ds = DigitSequenceDataset(cfg.dataset_path)

    # split dataset into training and validation
    train_set_len = int(len(ds) * cfg.train_split_fraction)
    train_ds, val_ds = random_split(ds, [train_set_len, len(ds) - train_set_len])

    # create dataloaders
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=8,
                              shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=8,
                            shuffle=False)

    # create rnn model
    rnn = DigitSumModel(VOCAB_SIZE, 128, VOCAB_SIZE)

    # define optimizer
    optim = torch.optim.Adam(rnn.parameters(), lr=cfg.learning_rate)

    # use custom loss function
    loss_func = OneHotCrossEntropyLoss()

    # train
    train(rnn, train_loader, val_loader, loss_func, optim, cfg.n_epochs, log_path, logger)
    logger.close()


if __name__ == "__main__":
    main()
