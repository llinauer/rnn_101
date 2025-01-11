"""
train.py

Train an RNN to predict the digit sum of a sequence
"""

from pathlib import Path
from time import gmtime, strftime

import einops
import hydra
import torch
import torch.nn.functional as F
from misc import check_accuracy
from model import DigitSumModel
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import (VOCAB_SIZE, DigitSequenceDataset, EqualLengthBatchSampler,
                  EqualLengthSampler)


class OneHotCrossEntropyLoss(nn.Module):
    """ Custom class for calculating Cross entropy with one-hot encoded targets """
    def forward(self, logits, one_hot_targets):
        log_probs = F.log_softmax(logits, dim=-1)
        cross_entropy = -einops.einsum(
            log_probs, one_hot_targets,
            "batch sequence vocab, batch sequence vocab -> batch sequence"
        )
        loss = cross_entropy.mean()
        return loss


def train(model, train_loader, val_loader, val_ds, loss_func, optimizer, scheduler, n_epochs,
          log_path, tb_logger, n_acc=10):
    """ Train function. Iterate over the batch_loader epochs times and train the model.
        Log metrics with tensorboard logger """

    # loop for n_epochs
    for epoch in range(n_epochs):

        train_loss = 0.0
        val_loss = 0.0
        train_grad_norm = 0.0
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
            input_logits, input_last_hidden = model(input_sequences)

            # pass the target sequences through the model to get further predictions
            target_logits, _ = model(target_sequences, h_0=input_last_hidden)

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

            grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)  # calc L2 norm
                    grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** 0.5
            train_grad_norm += grad_norm

            # accumulate training loss
            train_loss += loss.item()

            # print running training loss
            train_progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # log train loss and gradient norm
        tb_logger.add_scalar('Loss/train', avg_train_loss, epoch)
        tb_logger.add_scalar('Train grad norm', train_grad_norm / len(train_loader), epoch)

        # every 10 epochs, inform the user
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Average Loss: {avg_train_loss:.4f}")

        # validation loop, set model to eval mode
        model.eval()

        # create progress bar out of val_loader
        val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))
        for _, (input_sequences, target_sequences) in val_progress_bar:

            # pass input_sequences and target_sequences through the model
            input_logits, input_last_hidden = model(input_sequences)
            # pass the target sequences through the model to get further predictions
            target_logits, _ = model(target_sequences, h_0=input_last_hidden)

            # calculate loss like in training loop
            loss = loss_func(input_logits[:, -2:-1, :], target_sequences[:, 0:1, :])
            loss += loss_func(target_logits[:, :-1, :], target_sequences[:, 1:, :])
            val_loss += loss.item()

        scheduler.step(val_loss)

        # log the current learning reate
        for param_group in optimizer.param_groups:
            current_lr = param_group.get('lr', None)
            if current_lr:
                tb_logger.add_scalar("learning_rate", current_lr, epoch)
                break  # Assuming all param groups have the same learning rate

        avg_val_loss = val_loss / len(val_loader)
        # log val loss
        tb_logger.add_scalar("Loss/val", avg_val_loss, epoch)

        # check if model performance improved
        model_performance = -avg_val_loss
        if model_performance > best_model_performance:
            print(f"New best model performance. Saving model to {log_path/'best_model.pth'}")
            torch.save(model.cpu().state_dict(), log_path / "best_model.pth")

        # every n epochs, check the accuracy of the model outputs on the val dataset
        if epoch % n_acc == 0:
            print(f"Validation: Average Loss: {val_loss/len(val_loader):.4f}")
            val_acc = check_accuracy(model, val_ds)
            # log to tensorboard
            tb_logger.add_scalar("Val accuracy", val_acc*100, epoch)
            print(f"Validation: Accuracy: {val_acc*100:2f}")


@hydra.main(version_base=None, config_name="config", config_path=".")
def main(cfg: DictConfig) -> None:
    """ Main function """

    log_path = Path(cfg.train.log_path)

    # check if hidden size is given, is an integer and not too big or too small
    if not cfg.train.hidden_size:
        print("No train.hidden_size argument given, using default value of 128")
        hidden_size = 128

    if not isinstance(cfg.train.hidden_size, int) or not 16 <= cfg.train.hidden_size <= 256:
        print("train.hidden_size argument must be an integer between 16 and 256!")
        return
    hidden_size = cfg.train.hidden_size

    # check for model type
    if not cfg.train.model_type or not isinstance(cfg.train.model_type, str):
        print("Please provide valid model type with the train.model_type argument!"
              " Options: [rnn, lstm]")
        return
    model_type = cfg.train.model_type

    # check if weight decay should be used
    if cfg.train.weight_decay is not None:
        try:
            weight_decay = float(cfg.train.weight_decay)
        except ValueError:
            print("weight_decay config must be a float. Not using weight decay")
            weight_decay = 0.
    else:
        weight_decay = 0.

    # check if run name was given, if not create from current time
    if not cfg.train.run_name:
        run_name = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    else:
        run_name = cfg.train.run_name

    # create a path for each run
    log_path = log_path / run_name

    # create log_path if it does not exist
    if not log_path.exists():
        log_path.mkdir(parents=True, exist_ok=True)

    # check if dataset path is provided
    if not cfg.train.dataset_path:
        print("Please provide path to dataset with the 'train.dataset_path' argument")
        return

    # check if dataset path exists
    if not Path(cfg.train.dataset_path).exists():
        print(f"Dataset at {cfg.train.dataset_path} does not exist")
        return

    # create tensorboard logger
    logger = SummaryWriter(log_path)

    # load dataset
    ds = DigitSequenceDataset(cfg.train.dataset_path)

    # split dataset into training and validation
    train_set_len = int(len(ds) * cfg.train.train_split_fraction)
    train_ds, val_ds = random_split(ds, [train_set_len, len(ds) - train_set_len])

    # create custom sampler for equal sequence lengths in input and targets
    train_sampler = EqualLengthSampler(train_ds, shuffle=True)
    val_sampler = EqualLengthSampler(val_ds, shuffle=True)

    # create custom batch samplers
    train_batch_sampler = EqualLengthBatchSampler(train_sampler, cfg.train.batch_size)
    val_batch_sampler = EqualLengthBatchSampler(val_sampler, cfg.train.batch_size)

    # create dataloaders
    train_loader = DataLoader(train_ds, batch_sampler=train_batch_sampler, num_workers=8)
    val_loader = DataLoader(val_ds, batch_sampler=val_batch_sampler, num_workers=8)

    # create rnn model
    rnn = DigitSumModel(VOCAB_SIZE, hidden_size, VOCAB_SIZE, model_type=model_type)

    # define optimizer
    optim = torch.optim.Adam(rnn.parameters(), lr=cfg.train.learning_rate,
                             weight_decay=weight_decay)
    # define lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5,
                                                           patience=cfg.train.n_epochs // 20)

    # use custom loss function
    loss_func = OneHotCrossEntropyLoss()

    # train
    train(rnn, train_loader, val_loader, val_ds, loss_func, optim, scheduler, cfg.train.n_epochs,
          log_path, logger)
    logger.close()


if __name__ == "__main__":
    main()
