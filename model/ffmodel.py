import sys
import os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'vocabulary'))
import torch
import torch.nn as nn
import re

"""
DEVICE CONFIGURATION
"""
# check if a cuda-enabled gpu is available and use it, else use the cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
NEURAL NET
"""


# TODO batch normalization, TODO use ModuleList instead of hard coding the number of full layers
class NeuralNet(nn.Module):
    def __init__(self, window_size, num_full_layers, source_vocab_size, target_vocab_size):
        super(NeuralNet, self).__init__()

        # save parameters for printing and referencing
        self.window_size = window_size
        self.num_full_layers = num_full_layers
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size

        # embedded source and target window layers
        EMB_SIZE = 2000  # encoding level of our vocabulary
        self.emb_source = nn.Embedding(source_vocab_size, EMB_SIZE)
        self.emb_source_bn = nn.BatchNorm1d(2 * window_size + 1)
        self.emb_target = nn.Embedding(target_vocab_size, EMB_SIZE)
        self.emb_target_bn = nn.BatchNorm1d(window_size)

        # fully connected layer for source and target each
        OUT_SIZE = 300  # Size of the linear layers
        self.full_source = nn.Linear(EMB_SIZE, OUT_SIZE)  # fully connected layer
        # self.full_source_bn = nn.BatchNorm1d(2*window_size+1)
        self.full_target = nn.Linear(EMB_SIZE, OUT_SIZE)  # fully connected layer
        # self.full_target_bn = nn.BatchNorm1d(window_size)

        # concatenation layer only needed in forwarding

        # fully connected layers
        self.fc1 = nn.Linear(OUT_SIZE, OUT_SIZE)
        self.fc1_bn = nn.BatchNorm1d(OUT_SIZE)
        self.fc2 = nn.Linear(OUT_SIZE, OUT_SIZE)
        self.fc2_bn = nn.BatchNorm1d(OUT_SIZE)
        self.fc3 = nn.Linear(OUT_SIZE, OUT_SIZE)
        self.fc3_bn = nn.BatchNorm1d(OUT_SIZE)

        # Creates a list of layers
        # self.fc1.weight.data.fill_(1)
        # full_layers = []
        # for i in range(num_full_layers):
        #     full_layers.append(nn.Linear(OUT_SIZE, OUT_SIZE))
        #     # self.full.append(nn.BatchNorm1d(OUT_SIZE))
        # self.full = nn.ModuleList(full_layers)

        # projection layer after flattening
        self.project = nn.Linear(OUT_SIZE * (3 * window_size + 1), target_vocab_size)
        self.project_bn = nn.BatchNorm1d(target_vocab_size)
        # self.project.weight.data.fill_(0.05)

        # activation functions
        self.relu = nn.ReLU()
        self.tan_h = nn.Tanh()

    def forward(self, source_input, target_input, verbose=False):

        source_out = source_input
        target_out = target_input

        if verbose:
            print("Source input:")
            print(source_out)
            print("Target input:")
            print(target_out)

        # embedding layers
        source_out = self.emb_source(source_out)
        source_out = self.emb_source_bn(source_out)
        # source_out = self.relu(source_out)
        target_out = self.emb_target(target_out)  # changed self.source to self.emb_target
        target_out = self.emb_target_bn(target_out)
        # target_out = self.relu(target_out)

        if verbose:
            print("\n********************************************\n")
            print("Source after embedding layer:")
            print(source_out)
            print("Target after embedding layer:")
            print(target_out)

        # fully connected layer for source and target each
        source_out = self.full_source(source_out)
        source_out = self.relu(source_out)
        target_out = self.full_target(target_out)
        target_out = self.relu(target_out)

        if verbose:
            print("\n********************************************\n")
            print("Source after full source layer:")
            print(source_out)
            print("Target after full target layer:")
            print(target_out)

        # concatenation layer
        out = torch.cat((source_out, target_out), 1)
        out = self.relu(out)

        if verbose:
            print("\n********************************************\n")
            print("After concatenation layer:")
            print(out)

        if verbose:
            print("\n********************************************\n")
            print("After flattening:")
            print(out)

        # fully connected layers
        # for layer in self.full:
        #     layer.to(device)
        #     out = layer(out)
        #     out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)
        # out = self.fc2(out)
        # out = self.relu(out)
        # out = self.fc3(out)
        # out = self.relu(out)

        if verbose:
            print("\n********************************************\n")
            print("After full layers:")
            print(out)

        # flattening + projection layer
        out = torch.flatten(out, start_dim=1)
        out = self.project(out)
        out = self.project_bn(out)
        if verbose:
            print("\n********************************************\n")
            print("After projection layer:")
            print(out)

        return out

    # Used for printing the structure of the network:
    # - name of the layers
    # - input size and output size of the layers
    # - weight matrices
    def __str__(self):
        model_string = 'Printing our model...\n'
        # for t in self.parameters():
        #     print(t)

        model_string += '\nHyperparameters:\n'
        model_string += f'window size: {self.window_size}\n'
        model_string += f'number of full layers: {self.num_full_layers}\n'

        model_string += '\nLayer architecture:\n'
        model_string += "\n********************************************\n"
        model_string += 'Embedding layers:\n'
        model_string += f'Source embedding layer: {self.emb_source}\n'
        model_string += f'Target embedding layer: {self.emb_target}\n'
        model_string += "\n********************************************\n"
        model_string += 'Full layers for source and target each:\n'
        model_string += f'Full source layer: {self.full_source}\n'
        model_string += f'Full target layer: {self.full_target}\n'
        model_string += "\n********************************************\n"
        model_string += 'Concatenation layer\n'
        model_string += "\n********************************************\n"
        model_string += 'Full layers:\n'
        model_string += f'Full layer 1: {self.fc1}\n'
        # model_string += f'Full layer 2: {self.fc2}\n'
        model_string += "\n********************************************\n"
        # model_string += f'Full layer 3: {self.fc3}\n'
        # for layer in self.full:
        #     model_string += f'Full layer: {layer}\n'
        model_string += f'Projection layer: {self.project}\n'

        return model_string



"""
SAVING AND LOADING
"""
# save model
def save_model(model, model_name, optimizer_state_dict, epoch, batch, perplexity):
    file_name = model_name
    if epoch is not None: 
        file_name += f'[{epoch + 1}]'
    else:
        file_name += '[f]'
    if batch is not None:
        file_name += f'[{batch + 1}]'
    else:
        file_name += '[e_o_epoch]'
    file_name += '.pth'

    # check if folder exists, if not, then create one
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'saved_models', model_name)):
        os.makedirs(os.path.join(os.path.dirname(__file__), 'saved_models', model_name))

    file_path = os.path.join(os.path.dirname(__file__), 'saved_models', model_name, file_name)

    # create dictionary with items we want to save
    checkpoint = {
        'model': {
            'window_size': model.window_size,
            'num_full_layers': model.num_full_layers,
            'source_vocab_size': model.source_vocab_size,
            'target_vocab_size': model.target_vocab_size,
            'state_dict': model.state_dict(),  # can specify which model we want to save (if there are more than one)
        },
        'optimizer_state_dict': optimizer_state_dict,
        # can specify which optimizer we want to save (if there are more than one)
        'epoch': epoch,
        'batch': batch,
        'perplexity': perplexity
    }
    print(f"Saving our model to {file_path}")
    torch.save(checkpoint, file_path)

# load model based on its name
def load_model(model_name, whole_model_name=None, epoch=None, batch=None, model=None, optimizer=None, file_location=device):

    if(whole_model_name):
        file_name = whole_model_name
        print(f"loading model {model_name} with checkpoint file {whole_model_name} ...")

    else:
        print(f"loading model {model_name} for epoch {epoch} and batch {batch} ...")
        file_name = model_name
        if epoch is not None: 
            file_name += f'[{epoch + 1}]'
        else:
            file_name += '[f]'
        if batch is not None:
            file_name += f'[{batch + 1}]'
        else:
            file_name += '[f]'
        file_name += '.pth'

    checkpoint = torch.load(os.path.join(os.path.dirname(__file__), 'saved_models', model_name, file_name), map_location=device)

    # recreate model
    if not model:
        model = NeuralNet(window_size=checkpoint['model']['window_size'],
                          num_full_layers=checkpoint['model']['num_full_layers'],
                          source_vocab_size=checkpoint['model']['source_vocab_size'],
                          target_vocab_size=checkpoint['model']['target_vocab_size'])
    model.load_state_dict(checkpoint['model']['state_dict'])
    model.to(device)

    # optimizer
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # other information
    epoch = checkpoint['epoch']
    batch = checkpoint['batch']
    perplexity = checkpoint['perplexity']

    print(' -> done.')
    return {'model': model, 'optimizer': optimizer, 'epoch': epoch, 'batch': batch, 'perplexity': perplexity}

# wrapper for load_model() to load multiple checkpoints of one model for early stopping, thus only returning a list of models and not caring about the optizer and stuff...
def load_models(model_name):
    directory = os.path.join(os.path.dirname(__file__), 'saved_models', model_name)
    load_dicts = []
    dir_list = os.listdir(directory)
    ordered_dir = sort_dir(dir_list)
    print(ordered_dir)
    for model_file_name in ordered_dir:
        if model_file_name.startswith(model_name):
            load_dict = load_model(model_name=model_name, whole_model_name=model_file_name)
            load_dict['filename'] = model_file_name
            load_dicts.append(load_dict)
    return load_dicts

def sort_dir(dir):
    model_v = []
    order_version_list = []
    final_order_checkpoints = []
    for i in dir:
        version = [int(x) for x in re.findall(r'[0-9]+', i) if x != []][1:]
        model_v.append(version)
    sorted_list = sorted(model_v)

    for el in sorted_list:
        order_version_list.append(model_v.index(el))

    for i in order_version_list:
        final_order_checkpoints.append(dir[i])
    return final_order_checkpoints
