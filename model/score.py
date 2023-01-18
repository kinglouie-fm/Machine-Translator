import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'vocabulary'))
import dictionary
import ffmodel
import training
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import json

# check if a cuda-enabled gpu is available and use it, else use the cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Calculates the probability a sentence in the target file is the actual translation of the model of the source sentence
# Creates a dataset of each sentence and calculates the probability of each line in the dataset.
# The multiplication of every line in the dataset created
# checking model: the model we want to calculate our probability for
# dataset: the dataset from the given texts, is of type TranslatorDataset from ffmodel.py
def score(checking_model, dataset, prob_batchsize=300):
    # datasets = split_dataset_to_sentence_datasets(dataset)

    data_loader = DataLoader(dataset=dataset, batch_size=prob_batchsize)
    checking_model = checking_model.to(device)
    checking_model.eval()

    # probability we multiply the single word probabilities to and to return
    probability = 1

    # counter of which is the current sentence
    current_sentence = 1

    with torch.no_grad():
        for (source_input, target_input, labels) in data_loader:
            # load data to device
            source_input = source_input.to(device)
            target_input = target_input.to(device)
            labels = labels.to(device)

            # create the expected outputs and create a probability distribution using softmax
            outputs = checking_model(source_input, target_input)
            probabilities = outputs.softmax(dim=1)

            # create one_hot vectors from the label to find the corresponding word in the vocabulary
            one_hot = nn.functional.one_hot(labels, num_classes=checking_model.target_vocab_size).float()
            one_hot = torch.flatten(one_hot, start_dim=1)

            # goes through every line of the current batch and calculates the new probability for the current sentence
            for i in range(labels.size(0)):
                # max returns (value ,index)
                # returns the index of the current label within the target dictionary
                value, index = torch.max(one_hot[i], 0)
                added_probability = probabilities[i][index]

                # multiples the current probability
                probability = probability * added_probability

                if labels[i] == 1:  # A unique identifier of the end of a sentence
                    print(f"Probability of translation from sentence {current_sentence}: {probability}")
                    probability = 1
                    current_sentence += 1


if __name__ == '__main__':
    # parse the first 3 inputs into
    # 1. args.config_file_path
    # 2. args.source_file_path
    # 3. args.target_file_path
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file_path", help="(relative) path of the config file")
    parser.add_argument("source_file_name", help="name of the source file")
    parser.add_argument("target_file_name", help="name of the target file")
    args = parser.parse_args()

    # open config file and get parameters
    f = open(args.config_file_path)
    config_data = json.load(f)
    f.close()

    # dictionaries
    dict_source = dictionary.Dictionary(config_data['source_vocab_file_name'])
    dict_target = dictionary.Dictionary(config_data['target_vocab_file_name'])

    # create model
    blank_model = ffmodel.NeuralNet(
        window_size=config_data['window_size'], num_full_layers=config_data['num_full_layers'],
        source_vocab_size=len(dict_source), target_vocab_size=len(dict_target))

    # load model
    optimizer = torch.optim.Adam(blank_model.parameters(), lr=config_data["learning_rate"])
    # returns a dictionary. The key "model" is called to return the model
    # working_model = ffmodel.load_model(config_data["load_model_name"], epoch=config_data["epoch"] - 1)["model"]
    working_model = ffmodel.load_model(config_data["load_model_name"], epoch=config_data["epoch"] - 1)["model"]

    # creates the scoring data set based on the files given as parameters
    scoring_data_set = training.create_training_dataset(
        source_file_name=args.source_file_name, target_file_name=args.target_file_name,
        source_merge_operations_file_name=config_data['source_merge_operations_file_name'],
        target_merge_operations_file_name=config_data['target_merge_operations_file_name'],
        dict_source=dict_source, dict_target=dict_target, window_size=config_data['window_size'])

    # gives the probabilities of the translation of each sentence in the scoring data set
    score(working_model, scoring_data_set)
