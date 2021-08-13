import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer

from perceiver import Perceiver

BATCH_SIZE = 32
N_GPUS = 1


class LightningPerceiverSST2(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        num_classes,
        num_hidden_layers=8,
        hidden_size=256,
        latent_size=64,
        num_attention_heads=4,
    ):
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=hidden_size
        )
        self.perceiver = Perceiver(
            num_classes=num_classes if num_classes > 2 else 1,
            num_hidden_layers=num_hidden_layers,
            latent_size=latent_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, tokenized):

        embedded = self.embeddings(tokenized)
        outputs = self.perceiver(embedded)
        return outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.permute(1, 0))
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.permute(1, 0))
        loss = self.criterion(y_hat, y)
        return {"loss": loss, "labels": y, "logits": y_hat}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.permute(1, 0))
        loss = self.criterion(y_hat, y)
        return {"loss": loss, "labels": y, "logits": y_hat}

    def validation_epoch_end(self, outputs):
        # get predictions and probs from logits
        logits = torch.cat([x["logits"] for x in outputs])
        preds = torch.round(logits).detach().cpu().numpy()

        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        val_loss = torch.stack([x["loss"] for x in outputs]).mean()
        val_accuracy = accuracy_score(preds, labels)

        self.log("val_loss", val_loss)
        self.log("val_accuracy", val_accuracy)

        return val_loss

    def test_epoch_end(self, outputs):
        logits = torch.cat([x["logits"] for x in outputs])
        preds = torch.argmax(logits, axis=0).detach().cpu().numpy()

        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        test_loss = torch.stack([x["loss"] for x in outputs]).mean()
        test_accuracy = accuracy_score(preds, labels)

        self.log("test_loss", test_loss)
        self.log("test_accuracy", test_accuracy)

        return test_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


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
        train_inputs, padding=True, truncation=True, max_length=64, return_tensors="pt"
    )
    val_tokenized = tokenizer(
        val_inputs, padding=True, truncation=True, max_length=64, return_tensors="pt"
    )
    test_tokenized = tokenizer(
        test_inputs, padding=True, truncation=True, max_length=64, return_tensors="pt"
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
    lightning_model = LightningPerceiverSST2(vocab_size, num_classes=num_classes)
    trainer = pl.Trainer(max_epochs=10, gpus=N_GPUS)
    # trainer = pl.Trainer(max_epochs=1)
    trainer.fit(lightning_model, train_loader, val_loader)


if __name__ == "__main__":
    main()
