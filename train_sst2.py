import logging

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer

from perceiver import Perceiver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BATCH_SIZE = 32
NUM_GPUS = 1
NUM_EPOCHS = 5
NUM_HIDDEN_LAYERS = 4
HIDDEN_SIZE = 512
FEEDFORWARD_SIZE = 1024
SEQUENCE_LENGTH = 64
LATENT_SIZE = 256
NUM_ATTENTION_HEADS = 4
LEARNING_RATE = 1e-4

DEVICE = "cuda" if NUM_GPUS > 0 else "cpu"


class LightningPerceiverSST2(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        num_classes,
        num_hidden_layers=8,
        hidden_size=256,
        latent_size=64,
        feedforward_size=1024,
        max_seq_length=32,
        num_attention_heads=4,
    ):
        super().__init__()

        self.max_seq_length = max_seq_length

        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=hidden_size
        )
        self.position_embeddings = nn.Embedding(
            num_embeddings=max_seq_length, embedding_dim=hidden_size
        )
        self.perceiver = Perceiver(
            num_classes=num_classes if num_classes > 2 else 1,
            num_hidden_layers=num_hidden_layers,
            latent_size=latent_size,
            hidden_size=hidden_size,
            feedforward_size=feedforward_size,
            num_attention_heads=num_attention_heads,
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, tokenized):

        embedded = self.embeddings(tokenized)

        # add position embeddings
        positions = (
            torch.arange(tokenized.shape[1]).repeat(tokenized.shape[0], 1).to(DEVICE)
        )
        position_embedded = self.position_embeddings(positions)
        embedded = embedded + position_embedded

        outputs = self.perceiver(embedded.permute(1, 0, 2))
        return outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return {"loss": loss, "labels": y, "logits": y_hat}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return {"loss": loss, "labels": y, "logits": y_hat}

    def validation_epoch_end(self, outputs):
        # get predictions and probs from logits
        logits = torch.cat([x["logits"] for x in outputs])
        preds = torch.round(torch.sigmoid(logits)).detach().cpu().numpy().astype(int)
        labels = (
            torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy().astype(int)
        )
        val_loss = torch.stack([x["loss"] for x in outputs]).mean()
        val_accuracy = accuracy_score(preds, labels)

        logger.info(f"Val Loss : {val_loss} --- Val Accuracy : {val_accuracy}")
        self.log("val_loss", val_loss)
        self.log("val_accuracy", val_accuracy)

        return val_loss

    def test_epoch_end(self, outputs):
        logits = torch.cat([x["logits"] for x in outputs])
        preds = torch.round(torch.sigmoid(logits)).detach().cpu().numpy().astype(int)
        labels = (
            torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy().astype(int)
        )
        test_loss = torch.stack([x["loss"] for x in outputs]).mean()
        test_accuracy = accuracy_score(preds, labels)

        logger.info(f"Test Loss : {test_loss} --- Test Accuracy : {test_accuracy}")
        self.log("test_loss", test_loss)
        self.log("test_accuracy", test_accuracy)

        return test_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)


def main():

    # load dataset
    dataset = load_dataset("glue", "sst2")
    num_classes = len(list(set(dataset["train"]["label"])))
    print(dataset)

    # get text and labels
    train_inputs = dataset["train"]["sentence"]
    val_inputs = dataset["validation"]["sentence"]
    test_inputs = dataset["test"]["sentence"]

    train_labels = dataset["train"]["label"]
    val_labels = dataset["validation"]["label"]
    test_labels = dataset["test"]["label"]

    # tokenize
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    print(tokenizer)

    train_tokenized = tokenizer(
        train_inputs,
        padding=True,
        truncation=True,
        max_length=SEQUENCE_LENGTH,
        return_tensors="pt",
    )
    val_tokenized = tokenizer(
        val_inputs,
        padding=True,
        truncation=True,
        max_length=SEQUENCE_LENGTH,
        return_tensors="pt",
    )
    test_tokenized = tokenizer(
        test_inputs,
        padding=True,
        truncation=True,
        max_length=SEQUENCE_LENGTH,
        return_tensors="pt",
    )

    # create dataloaders
    train_dataset = TensorDataset(
        train_tokenized["input_ids"], torch.FloatTensor(train_labels)
    )
    val_dataset = TensorDataset(
        val_tokenized["input_ids"], torch.FloatTensor(val_labels)
    )
    test_dataset = TensorDataset(
        test_tokenized["input_ids"], torch.FloatTensor(test_labels)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # training
    lightning_model = LightningPerceiverSST2(
        vocab_size,
        num_classes=num_classes,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
        feedforward_size=FEEDFORWARD_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        max_seq_length=SEQUENCE_LENGTH,
    )
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS, gpus=NUM_GPUS)
    # trainer = pl.Trainer(max_epochs=1)
    trainer.fit(lightning_model, train_loader, val_loader)


if __name__ == "__main__":
    main()
