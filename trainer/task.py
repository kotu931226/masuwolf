from statistics import mean
import argparse
import datetime
import sys
import os
path_of_here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_of_here, '../'))
import logging
logging.basicConfig(level=logging.INFO)
import torch
from torch import nn
from trainer.utils import DataOperat
from trainer.model import ClassifyTransformer
from trainer.config import ModelParams, DataPath

# torch.backends.cudnn.benchmark = True

logging.info('start script')
logging.info('use {}'.format(ModelParams.device))

# set arguments data
parser = argparse.ArgumentParser()
parser.add_argument('--archive-path', help='absolute dir path',
                    default=DataPath.archive_path)
parser.add_argument('--input-data', help='absolute file path',
                    default=DataPath.input_data_path)
parser.add_argument('--label-data', help='absolute file path',
                    default=DataPath.label_data_path)
parser.add_argument('--input-model', help='absolute dir path',
                    default=None)
parser.add_argument('--epochs', help='set epochs',
                    type=int, default=ModelParams.n_epochs)
parser.add_argument('--job-dir')
args = parser.parse_args()
DataPath.archive_path = args.archive_path
DataPath.input_data_path = args.input_data
DataPath.label_data_path = args.label_data
DataPath.input_model_path = args.input_model
ModelParams.n_epoch = args.epochs

logging.info('config loaded')

# create model
model = ClassifyTransformer(
    ModelParams.ids_size, ModelParams.n_classes, ModelParams.d_model,
    ModelParams.d_ff, ModelParams.N, ModelParams.n_heads,
    device=ModelParams.device
)
model = model.to(ModelParams.device)
optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=0.003)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
data_set = DataOperat.create_data_set(
    DataPath.input_data_path, DataPath.label_data_path, device=ModelParams.device
)
data_set = data_set[:3000*5] # To change size for small data_set
dev_idx = len(data_set)*7//8
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
# loss_fn = nn.NLLLoss(ignore_index=0)

if DataPath.input_model_path:
    DataOperat.load_torch_model(DataPath.input_model_path, model)

# start train def
def train(model, data, optimizer, n_epoch, batch_size, dev_data=None):
    for epochs in range(1, n_epoch+1):
        model.train()
        correct = 0
        count = 0
        losses = []
        logging.info('-----Epoch: {}'.format(epochs))
        gen_batch_data = DataOperat.gen_batch_data(data, batch_size)
        for i, batch_data in enumerate(gen_batch_data):
            preds = model(batch_data[0])
            loss = loss_fn(preds, batch_data[1])
            losses.append(loss.item())
            _, pred_ids = torch.max(preds, 1)
            correct += torch.sum(pred_ids == batch_data[1]).item()
            count += batch_size
            if i % 100 == 0:
                logging.info('Epoch {} iteration {} train_acc {:.5f} train_loss {:.5f}'
                            .format(epochs, i, correct / count, mean(losses)))
            model.zero_grad()
            loss.backward()
            optimizer.step()
        test(model, dev_data, batch_size, correct / count, mean(losses))
        if epochs % 20 == 0:
            DataOperat.save_torch_model(DataPath.emit_temp_model_path(epochs), model)

def test(model, dev_data, batch_size, train_acc, train_loss):
    model.eval()
    correct = 0
    count = 0
    losses = []
    gen_batch_data = DataOperat.gen_batch_data(dev_data, batch_size)
    for batch_data in gen_batch_data:
        preds = model(batch_data[0])
        loss = loss_fn(preds, batch_data[1])
        losses.append(loss.item())
        _, pred_ids = torch.max(preds, 1)
        correct += torch.sum(pred_ids == batch_data[1]).item()
        count += batch_size
    logging.info('-----Test Result-----')
    logging.info('train_accuracy:{}'.format(train_acc))
    logging.info('val_accuracy:  {}'.format(correct / count))
    logging.info('train_loss:{}'.format(train_loss))
    logging.info('val_loss:  {}'.format(mean(losses)))
    DataOperat.add_csv(DataPath.emit_losses_path(),
        [[
            train_acc,
            correct / count,
            train_loss,
            mean(losses),
            datetime.datetime.now()
        ]])

train_data = data_set[:dev_idx]
dev_data = data_set[dev_idx:]

train(
    model, train_data, optimizer,
    ModelParams.n_epoch, ModelParams.batch_size, dev_data=dev_data
)

DataOperat.save_torch_model(DataPath.emit_output_model_path(), model)
logging.info('end train')
