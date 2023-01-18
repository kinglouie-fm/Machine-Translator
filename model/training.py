import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import argparse
import json

import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'vocabulary'))
import bpe
import dataset_preparation
import dictionary
import ffmodel



"""
DATASET
"""
class TrainingDataset(Dataset):

    def __init__(self, data, source_vocab_size, target_vocab_size):
        self.data = data
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        line = self.data[index]
        source = torch.tensor(line[0])
        target = torch.tensor(line[1])
        label = torch.tensor(line[2])
        return source, target, label

    # we can call len(dataset) to return the size
    def __len__(self):
        return len(self.data)

def create_training_dataset(source_file_name, target_file_name, source_merge_operations_file_name,
                            target_merge_operations_file_name, dict_source, dict_target, window_size=2):
    # apply bpe on the files (by giving the file name)
    source_sentences = bpe.get_text_with_applied_bpe(source_file_name, source_merge_operations_file_name)
    target_sentences = bpe.get_text_with_applied_bpe(target_file_name, target_merge_operations_file_name)

    source_sentences = dict_source.filter_text_by_vocabulary(source_sentences)
    target_sentences = dict_target.filter_text_by_vocabulary(target_sentences)

    # prepare the data further by calling the function dataset_preparation which will create S, T, L
    # replace our strings with indices according to our dictionaries
    data = dataset_preparation.training_preparation(source_sentences, target_sentences, dict_source, dict_target,
                                                    window_size)

    # create and return dataset
    source_vocab_size = len(dict_source)
    target_vocab_size = len(dict_target)
    data_set = TrainingDataset(data, source_vocab_size, target_vocab_size)

    return data_set



"""
TRAINING + DEVELOPMENT LOOPS
"""
def training(model, train_data_set, dev_data_set, num_epochs, learning_rate, half_learning_rate, batch_size,
             checkpoints=[1, None], load_model_name=None, save_model_name='model', verbose=False):
    # create data loader
    data_loader = DataLoader(dataset=train_data_set, batch_size=batch_size, shuffle=True)

    # move model to device and print it (hyperparameters and network structure)
    model = model.to(ffmodel.device)
    model.train()
    print(model)

    # set loss function
    loss_fn = nn.CrossEntropyLoss()

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # total steps per epoch (for printing purposes)
    n_total_steps = len(data_loader)

    perp = 0

    # load model and corresponding data if desired
    if load_model_name:
        model, optimizer, start_epoch, perp = ffmodel.load_model(model_name=load_model_name, model=model, optimizer=optimizer, file_location=ffmodel.device)
        print(f"Loaded epoch {start_epoch} with perplexity: {perp}")
    else:
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):

        # loop over our batches
        sum_loss = []
        for i, (source_input, target_input, labels) in enumerate(data_loader):

            # load data to GPU
            source_input = source_input.to(ffmodel.device)
            target_input = target_input.to(ffmodel.device)
            labels = labels.to(ffmodel.device)

            # forward pass
            outputs = model(source_input, target_input, verbose=verbose)
            labels = nn.functional.one_hot(labels, num_classes=model.target_vocab_size).float()
            labels = torch.flatten(labels, start_dim=1)
            loss = loss_fn(outputs, labels)

            # tries to half the learning rate every batch_size steps
            # halves it if the std of sum loss is smaller than the range of sum loss divided by 8
            if half_learning_rate:
                sum_loss.append(loss.item())
                if (i + 1) % batch_size == 0:
                    standard_deviation = np.std(sum_loss)
                    print("STANDARD DEVIATION: ", standard_deviation)
                    if standard_deviation <= (max(sum_loss) - min(sum_loss)) * 0.125:
                        learning_rate /= 2
                        sum_loss = []
                        print("lowered the learning rate to ", learning_rate)

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            perp = perplexity(loss.item())

            # save our model every n batches
            if checkpoints[1]:
                if (i + 1) % checkpoints[1] == 0:
                    ffmodel.save_model(model, save_model_name, optimizer.state_dict(), epoch, i, perp)

            if (i + 1) % 300 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Perplexity: {perp:.4f}')

        # save our model every n epochs
        if checkpoints[0]:
            if (epoch + 1) % checkpoints[0] == 0:
                ffmodel.save_model(model, save_model_name, optimizer.state_dict(), epoch, len(data_loader), perp)

        # call development
        development(model, dev_data_set, batch_size, loss_fn)

    # save our model a last time at the end
    # ffmodel.save_model(model, save_model_name, optimizer.state_dict(), num_epochs - 1, len(data_loader), perp)

def development(model, data_set, batch_size, loss_fn):
    # DataLoader
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size)
    model = model.to(ffmodel.device)
    model.eval()

    # in test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        n_correct = 0
        running_loss = 0
        for (source_input, target_input, labels) in data_loader:

            # load data to device
            source_input = source_input.to(ffmodel.device)
            target_input = target_input.to(ffmodel.device)
            labels = labels.to(ffmodel.device)

            # forward pass
            outputs = model(source_input, target_input)
            labels = nn.functional.one_hot(labels, num_classes=model.target_vocab_size).float()
            labels = torch.flatten(labels, start_dim=1)

            # max returns (value ,index)
            _, predicted = torch.max(outputs.softmax(dim=1), 1)
            pred = nn.functional.one_hot(predicted, num_classes=model.target_vocab_size).float()

            # compare predicted max value with label value
            # -> if they are equal, we increase n_correct by 1 to determine the accuracy in the end
            for i in range(labels.size(0)):

                _, pred_max = torch.max(pred[i], 0)
                _, lab_max = torch.max(labels[i], 0)
                #  print('Pred and lab: ', pred_max.item(), lab_max.item())
                if pred_max.item() == lab_max.item():
                    n_correct += 1
                    # print('Correct: ', n_correct)

            # loss
            running_loss += loss_fn(outputs, labels)

        acc = accuracy(n_correct, len(data_loader.dataset))
        perp = perplexity(running_loss / len(data_loader))

        print(f'DEVELOPMENT yields: Accuracy: {acc:.4f}%, Perplexity: {perp:.4f}')

    model.train()



"""
METRICS
"""
def accuracy(n_correct, n_samples):
    return 100.0 * n_correct / n_samples


def perplexity(crossentropy):
    return math.e ** crossentropy



"""
TESTING
"""
if __name__ == '__main__':
    # get config file as command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file_path", help="(relative) path of the config file")
    args = parser.parse_args()

    # open config file and get parameters
    # checkpoints: [x,y] describes that we want to save our model every x epochs and in every epoch every y batches
    f = open(args.config_file_path)
    config_data = json.load(f)

    # dictionaries
    dict_source = dictionary.Dictionary(config_data['train_source_file_name'],
                                        config_data['source_merge_operations_file_name'])
    dict_target = dictionary.Dictionary(config_data['train_target_file_name'],
                                        config_data['target_merge_operations_file_name'])

    # create model
    model = ffmodel.NeuralNet(window_size=config_data['window_size'], num_full_layers=config_data['num_full_layers'],
                      source_vocab_size=len(dict_source), target_vocab_size=len(dict_target))

    """
    TRAINING AND DEVELOPMENT
    """
    # create training dataset
    train_data_set = create_training_dataset(
        source_file_name=config_data['train_source_file_name'], target_file_name=config_data['train_target_file_name'],
        source_merge_operations_file_name=config_data['source_merge_operations_file_name'],
        target_merge_operations_file_name=config_data['target_merge_operations_file_name'],
        dict_source=dict_source, dict_target=dict_target, window_size=config_data['window_size'])

    # create development dataset
    dev_data_set = create_training_dataset(
        source_file_name=config_data['dev_source_file_name'], target_file_name=config_data['dev_target_file_name'],
        source_merge_operations_file_name=config_data['source_merge_operations_file_name'],
        target_merge_operations_file_name=config_data['source_merge_operations_file_name'],
        dict_source=dict_source, dict_target=dict_target, window_size=config_data['window_size'])

    # perform training using the dataset and the model
    training(model=model, train_data_set=train_data_set, dev_data_set=dev_data_set,
             num_epochs=config_data['num_epochs'], learning_rate=config_data['learning_rate'],
             half_learning_rate=config_data['half_learning_rate'], batch_size=config_data['batch_size'],
             checkpoints=config_data['checkpoints'], load_model_name=config_data['load_model_name'],
             save_model_name=config_data['save_model_name'], verbose=config_data['verbose'])
