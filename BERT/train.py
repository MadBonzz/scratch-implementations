import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import SequentialLR, LinearLR, LambdaLR
from bert import BERT
from dataset import HindiData, DataCollatorWithDynamicPadding
from utils import MLMLoss, EarlyStopping

torch.manual_seed(42)

train_batch_size = 80
test_batch_size = 32
n_epochs = 100
warmup_steps = 1
lr = 1e-4
weight_decay = 0.01

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_path = "C:\\Users\\shour\\OneDrive - vit.ac.in\\torch-implementations\\train.txt"
test_path = "C:\\Users\\shour\\OneDrive - vit.ac.in\\torch-implementations\\test.txt"

with open(train_path, 'r', encoding='utf-8') as file:
    train_sentences = file.readlines()

with open(test_path, 'r', encoding='utf-8') as file:
    test_sentences = file.readlines()

train_sentences = [sentence.strip() for sentence in train_sentences]
test_sentences = [sentence.strip() for sentence in test_sentences]

tokenizer_path = "bert-tiny-hindi"
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

train_dataset = HindiData(train_sentences, tokenizer, 0.15)
test_dataset = HindiData(test_sentences, tokenizer, 0.15)

data_collator = DataCollatorWithDynamicPadding(tokenizer)

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=data_collator)
test_loader = DataLoader(test_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=data_collator)


tiny_model = BERT(6, 512, 8, 0.1, 200, len(tokenizer), 10000).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(tiny_model.parameters(), lr=lr, weight_decay=weight_decay)
es = EarlyStopping(tolerance=20, min_delta=0.001)
warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)

def linear_decay_lambda(current_step):
    if current_step < warmup_steps:
        return 1.0
    return max(0.0, (n_epochs - current_step) / (n_epochs - warmup_steps))

decay_scheduler = LambdaLR(optimizer, lr_lambda=linear_decay_lambda)

scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps])

log_file = "training_metrics.txt"
with open(log_file, "w") as f:  # Creates/overwrites file
    f.write("epoch,train_loss,test_loss,learning_rate\n")

train_losses = []
test_losses = []
lr_history = []
for i in range(n_epochs):
    train_loss = 0
    test_loss = 0
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        output = tiny_model(input_ids)
        loss = MLMLoss(output, labels, input_ids != labels, criterion)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        output = tiny_model(input_ids)
        loss = MLMLoss(output, labels, input_ids != labels, criterion)
        test_loss += loss.item()
    train_loss /= len(train_loader)
    test_loss /= len(test_loader)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    lr_history.append(optimizer.param_groups[0]["lr"])
    print(f"Epoch : {i}, Train Loss : {train_loss}, Test Loss : {test_loss}, Learning Rate: {scheduler.get_last_lr()[0]}")
    scheduler.step()
    with open(log_file, "a") as f:
        f.write(f"{i},{train_loss:.4f},{test_loss:.4f},{lr_history[-1]:.6f}\n")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(lr_history, color="green")
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    
    plt.tight_layout()
    plt.savefig("training_progress.png")
    plt.close()
    es(test_loss, tiny_model.state_dict(), optimizer.state_dict(), i)
    if es.early_stop:
        break

tiny_model.load_state_dict(es.model_dict)
torch.save(tiny_model.state_dict(), 'bert.pth')