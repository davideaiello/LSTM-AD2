import arguments
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import logging
from tqdm import tqdm
import evaluate
import dataset
from model import LSTMAD

logging.basicConfig(level=logging.INFO, format='%(message)s')

args = arguments.parse_arguments()   
logging.info(f"Arguments: {vars(args)}")      

X_train = dataset.read_folder_normal(args.dataset_folder, args.frequency)
X_train, pipeline = dataset.preprocess_data(X_train)
Dataloader_train, DataLoader_val = dataset.split_data(X_train, args.train_split)

model = LSTMAD(X_train.shape[1], args.lstm_layers, args.window_size, args.prediction_length)

optimizer = Adam(model.parameters(), lr=args.lr)
if args.scheduler:
    scheduler = MultiStepLR(optimizer, milestones=[3], gamma=0.01)

if args.device == 'cuda':
    model = model.to('cuda')

criterion = nn.MSELoss()


for epoch in range(args.epochs_num):
    epoch_losses = np.zeros((0, 1), dtype=np.float32)
    model.train()
    for x, y in tqdm(Dataloader_train):
        if args.device == 'cuda':
            x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        loss = model.predict(x, y, criterion)
        loss.backward()
        optimizer.step()
        epoch_losses = np.append(epoch_losses, loss.item())

    if args.scheduler:
        scheduler.step()

    model.eval()
    valid_losses = []
    for x, y in tqdm(DataLoader_val):
        if args.device == 'cuda':
            x, y = x.cuda(), y.cuda()
        valid_losses.append(loss.item())
    validation_loss = sum(valid_losses)
    logging.info(
            f"Epoch {epoch+1}: Training Loss : {epoch_losses.mean():.4f} \t "
            f"Validation Loss {validation_loss / len(DataLoader_val)}"
        )
logging.info("Estimating normal distribution...")

errors = []
for x, y in tqdm(DataLoader_val):
    y_preds = []
    if args.device == 'cuda':
        x, y = x.cuda(), y.cuda()
    for p in range(args.prediction_length):
        x_w = x[:, p]                           # per ogni sample prendo ogni finestra possibile
        y_p = model.forward(x_w)                # calcolo la y di ongi finestra (che avrà una lunghezza di l samples)
        y_p = y_p[:, y_p.shape[1]-(X_train.shape[1])*(p+1):y_p.shape[1]-(X_train.shape[1])*p]       # per ongi y, prendo solo il sample di 
                                                                                                    # posizione l-esima relativo all'l-esima 
                                                                                                    # finestra
        y_preds.append(y_p)
    y_preds = torch.cat(y_preds, dim=1)
    e = torch.abs(y - y_preds)
    errors.append(e)
model.anomaly_scorer.find_distribution(torch.cat(errors))
model.save()


logging.info("Testing the model...")
evaluate.evaluation(model, pipeline)


