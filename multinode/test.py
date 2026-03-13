#!/usr/bin/env python
# coding: utf-8

# 20 newsgroup text classification with DistilBERT finetuning

import os
import sys
from datetime import datetime

import numpy as np
import torch
from packaging.version import Version as LV
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

torch.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using PyTorch version:", torch.__version__, " Device:", device, flush=True)
assert LV(torch.__version__) >= LV("1.0.0")


def correct(output, target):
    predicted = output.argmax(1)
    correct_ones = (predicted == target).type(torch.float)
    return correct_ones.sum().item()


def train(data_loader, model, scheduler, optimizer):
    model.train()

    num_batches = 0
    num_items = 0
    total_loss = 0.0
    total_correct = 0

    for batch_idx, (input_ids, input_mask, labels) in enumerate(data_loader):
        if batch_idx == 0:
            print("Entered first training batch", flush=True)

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            labels=labels,
        )

        loss = output.loss
        logits = output.logits

        total_loss += loss.item()
        num_batches += 1
        total_correct += correct(logits, labels)
        num_items += len(labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

    return {
        "loss": total_loss / num_batches,
        "accuracy": total_correct / num_items,
    }


def test(data_loader, model):
    model.eval()

    num_batches = 0
    num_items = 0
    total_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        for input_ids, input_mask, labels in data_loader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            labels = labels.to(device)

            output = model(
                input_ids=input_ids,
                attention_mask=input_mask,
                labels=labels,
            )

            loss = output.loss
            logits = output.logits

            total_loss += loss.item()
            num_batches += 1
            total_correct += correct(logits, labels)
            num_items += len(labels)

    return {
        "loss": total_loss / num_batches,
        "accuracy": total_correct / num_items,
    }


def log_measures(ret, log, prefix, epoch):
    for key, value in ret.items():
        log.add_scalar(prefix + "_" + key, value, epoch)


def main():
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir = os.path.join(os.getcwd(), "logs", "20ng-distilbert-" + time_str)
    print("TensorBoard log directory:", logdir, flush=True)
    os.makedirs(logdir, exist_ok=True)
    log = SummaryWriter(logdir)

    datapath = os.getenv("DATADIR")
    if datapath is None:
        print("Please set DATADIR environment variable!", flush=True)
        sys.exit(1)

    text_data_dir = os.path.join(datapath, "20_newsgroup")

    print("Processing text dataset", flush=True)

    texts = []
    labels_index = {}
    labels = []

    for name in sorted(os.listdir(text_data_dir)):
        path = os.path.join(text_data_dir, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            print("-", name, label_id, flush=True)

            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    args = {} if sys.version_info < (3,) else {"encoding": "latin-1"}
                    with open(fpath, **args) as f:
                        t = f.read()
                        i = t.find("\n\n")
                        if 0 < i:
                            t = t[i:]
                        texts.append(t)
                    labels.append(label_id)

    print("Found %s texts." % len(texts), flush=True)

    TEST_SET = 4000

    (
        sentences_train,
        sentences_test,
        labels_train,
        labels_test,
    ) = train_test_split(
        texts,
        labels,
        test_size=TEST_SET,
        shuffle=True,
        random_state=42,
    )

    print("Length of training texts:", len(sentences_train), flush=True)
    print("Length of training labels:", len(labels_train), flush=True)
    print("Length of test texts:", len(sentences_test), flush=True)
    print("Length of test labels:", len(labels_test), flush=True)

    print("The first training sentence:", flush=True)
    print(sentences_train[0], "LABEL:", labels_train[0], flush=True)

    print("Initializing DistilBertTokenizer", flush=True)

    MODEL_NAME = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    MAX_LEN_TRAIN = 128
    MAX_LEN_TEST = 512

    print("Tokenizing training set...", flush=True)
    train_encodings = tokenizer(
        sentences_train,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN_TRAIN,
        return_attention_mask=True,
    )

    print("Tokenizing test set...", flush=True)
    test_encodings = tokenizer(
        sentences_test,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN_TEST,
        return_attention_mask=True,
    )

    print("The token ids of the first training sentence:", flush=True)
    print(train_encodings["input_ids"][0], flush=True)

    ids_train = np.array(train_encodings["input_ids"])
    amasks_train = np.array(train_encodings["attention_mask"])

    ids_test = np.array(test_encodings["input_ids"])
    amasks_test = np.array(test_encodings["attention_mask"])

    (
        train_inputs,
        validation_inputs,
        train_labels,
        validation_labels,
    ) = train_test_split(
        ids_train,
        labels_train,
        random_state=42,
        test_size=0.1,
    )

    (
        train_masks,
        validation_masks,
        _,
        _,
    ) = train_test_split(
        amasks_train,
        ids_train,
        random_state=42,
        test_size=0.1,
    )

    train_inputs = torch.tensor(train_inputs, dtype=torch.long)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    train_masks = torch.tensor(train_masks, dtype=torch.long)

    validation_inputs = torch.tensor(validation_inputs, dtype=torch.long)
    validation_labels = torch.tensor(validation_labels, dtype=torch.long)
    validation_masks = torch.tensor(validation_masks, dtype=torch.long)

    test_inputs = torch.tensor(ids_test, dtype=torch.long)
    test_labels = torch.tensor(labels_test, dtype=torch.long)
    test_masks = torch.tensor(amasks_test, dtype=torch.long)

    BATCH_SIZE = 16

    print("Train: ", end="", flush=True)
    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )
    print(len(train_dataset), "messages", flush=True)

    print("Validation: ", end="", flush=True)
    validation_dataset = TensorDataset(
        validation_inputs, validation_masks, validation_labels
    )
    validation_sampler = SequentialSampler(validation_dataset)
    validation_loader = DataLoader(
        validation_dataset,
        sampler=validation_sampler,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )
    print(len(validation_dataset), "messages", flush=True)

    print("Test: ", end="", flush=True)
    test_dataset = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )
    print(len(test_dataset), "messages", flush=True)

    print("Initializing DistilBertForSequenceClassification", flush=True)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=20,
    )
    model = model.to(device)

    print("Testing GPU with a small tensor op...", flush=True)
    x = torch.randn(2, 2).to(device)
    y = x @ x
    print("GPU test passed:", y.shape, flush=True)

    num_epochs = 4
    weight_decay = 0.01
    lr = 2e-5
    warmup_steps = max(1, int(0.2 * len(train_loader)))

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    print("Creating optimizer...", flush=True)
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)

    print("Creating scheduler...", flush=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=len(train_loader) * num_epochs,
    )

    print("About to fetch first training batch...", flush=True)
    first_batch = next(iter(train_loader))
    print("Fetched first training batch.", flush=True)
    print(
        "First batch shapes:",
        first_batch[0].shape,
        first_batch[1].shape,
        first_batch[2].shape,
        flush=True,
    )

    start_time = datetime.now()
    print("About to start training loop...", flush=True)

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}", flush=True)

        train_ret = train(train_loader, model, scheduler, optimizer)
        log_measures(train_ret, log, "train", epoch)

        print(f"Finished training epoch {epoch + 1}", flush=True)

        val_ret = test(validation_loader, model)
        log_measures(val_ret, log, "val", epoch)

        print(f"Finished validation epoch {epoch + 1}", flush=True)
        print(
            f"Epoch {epoch+1}: "
            f"train loss: {train_ret['loss']:.6f} "
            f"train accuracy: {train_ret['accuracy']:.2%}, "
            f"val loss: {val_ret['loss']:.6f}, "
            f"val accuracy: {val_ret['accuracy']:.2%}",
            flush=True,
        )

    end_time = datetime.now()
    print("Total training time: {}.".format(end_time - start_time), flush=True)

    ret = test(test_loader, model)
    print(
        f"\nTesting: loss: {ret['loss']:.6f} accuracy: {ret['accuracy']:.2%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
