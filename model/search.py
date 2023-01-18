import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'vocabulary'))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'metrics'))
import ffmodel as nnet
import bpe
import dictionary
from bleu import bleu
from dataset_preparation import alignment
import argparse
import json
import torch
import math

AVERAGE_SOURCE_SENTENCE_LENGTH = 18
AVERAGE_TARGET_SENTENCE_LENGTH = 19
TOLERANCE = 0.2  # defines how much longer than the average target sentence length we want to search

'''
SEARCH ALGORITHMS PER SENTENCE
'''


def greedy_search(model, dict_source, dict_target, source_sentence_indices):
    # perform search to predict a target sentence
    target_sentence = []
    with torch.no_grad():

        target_sentence_indices = [dict_target.get_index_by_word('<s>')] * model.window_size

        # in each loop iteration we determine one target word, the number of loop iterations is however limited based on the expected length in cases we just do not determine the eos symbol
        max_target_length = (len(source_sentence_indices) - model.window_size) * round(
            (AVERAGE_TARGET_SENTENCE_LENGTH / AVERAGE_SOURCE_SENTENCE_LENGTH) * (1 + TOLERANCE))
        for i in range(1, max_target_length + 1):
            # the last window size elements are our target
            target_input = target_sentence_indices[-model.window_size:]
            target_input = torch.tensor([target_input], dtype=torch.int).to(nnet.device)
            # in this case out list already contains the start of sequence symbols, however with an alignment of 1 we would like to express that we want the first real word. Thus, we have to add the window size-1 to our alignment function to get the alignment value (first real word) and then get the range from alignment value - window size to alignment value + window size
            source_alignment = alignment(AVERAGE_TARGET_SENTENCE_LENGTH, AVERAGE_SOURCE_SENTENCE_LENGTH,
                                         i) - 1 + model.window_size
            # because we did not yet add </s> to the source input (because in the search we do not know how many of them we are going to need), we then just fill up the list with </s> if it is too short
            source_input = source_sentence_indices[
                           source_alignment - model.window_size: source_alignment + model.window_size + 1]
            source_input.extend(
                [dict_source.get_index_by_word('</s>')] * ((2 * model.window_size + 1) - len(source_input)))
            source_input = torch.tensor([source_input], dtype=torch.int).to(nnet.device)

            output = model(source_input, target_input)
            pred_label = torch.argmax(output).tolist()

            # if the end of sequence symbol is determined, we can quit the loop earlier
            if pred_label == dict_target.get_index_by_word('</s>'):
                break
            else:
                target_sentence_indices.append(pred_label)
                target_sentence.append(dict_target.get_word_by_index(pred_label))

    return [
        target_sentence]  # return it wrapped in a list because we do not want to make case distinctions for greedy and beam search when unpacking later


def beam_search(model, dict_source, dict_target, source_sentence_indices, beam_size, n_best):
    # perform search to predict a target sentence
    target_sentences = []
    with torch.no_grad():

        # instantiate a list of lists of the target indices yet determined.
        # initialized with only one element (only one list of start of sequence symbols)
        #    because we do not yet have multiple hypotheses to work with. That changes
        #    after the first iteration of the while loop when the first labels have been
        #    predicted. Then we always have 'beam_size' list elements
        current_hypotheses = [{
            'indices': [dict_target.get_index_by_word('<s>')] * model.window_size,
            'value': [],
            'pr': math.log(1),
            'final': False
        }]

        # perform multiple search steps until no new 
        end_of_target_sentence_reached = False
        i = 1
        max_target_length = (len(source_sentence_indices) - model.window_size) * round(
            (AVERAGE_TARGET_SENTENCE_LENGTH / AVERAGE_SOURCE_SENTENCE_LENGTH) * (1 + TOLERANCE))
        while (not end_of_target_sentence_reached):

            new_hypotheses = []
            end_of_target_sentence_reached = True
            for current_hypothesis in current_hypotheses:

                if current_hypothesis['final'] or len(current_hypothesis['value']) >= AVERAGE_TARGET_SENTENCE_LENGTH * (
                        1 + TOLERANCE):
                    new_hypotheses.append(current_hypothesis)

                else:
                    end_of_target_sentence_reached = False
                    # the last window size elements of our current hypothesis are our target input
                    target_input = current_hypothesis['indices'][-model.window_size:]
                    target_input = torch.tensor([target_input], dtype=torch.int).to(nnet.device)

                    # determining the source input like in greedy search
                    source_alignment = alignment(AVERAGE_TARGET_SENTENCE_LENGTH, AVERAGE_SOURCE_SENTENCE_LENGTH,
                                                 i) - 1 + model.window_size
                    source_input = source_sentence_indices[
                                   source_alignment - model.window_size: source_alignment + model.window_size + 1]
                    source_input.extend(
                        [dict_source.get_index_by_word('</s>')] * ((2 * model.window_size + 1) - len(source_input)))
                    source_input = torch.tensor([source_input], dtype=torch.int).to(nnet.device)

                    # now we get the probabilities and values of the top beam_size outputs from our model
                    output = model(source_input, target_input).softmax(dim=1)
                    probs, pred_labels = torch.topk(output, beam_size)

                    # now add all predicted labels and the corresponding accumulated probabilities to our new target sentences indices list
                    for pred_label, pred_pr in zip(pred_labels.tolist()[0], probs.tolist()[0]):

                        # if search yields the end of sequence symbol we do not want to extend the hypothesis anymore in following iterations, thus we add the final attribute
                        new_hypothesis = {
                            'indices': current_hypothesis['indices'] + [pred_label],
                            'value': current_hypothesis['value'] + [dict_target.get_word_by_index(pred_label)],
                            'pr': (current_hypothesis['pr'] + math.log(pred_pr)),
                            'final': False
                        }

                        # we nevertheless append the end of sequence symbol because if we did not do that we would run into special cases where we divide by zero when determining the normalized probabilites further down
                        if pred_label == dict_target.get_index_by_word('</s>'):
                            new_hypothesis['final'] = True

                        new_hypotheses.append(new_hypothesis)

            # determine the top beam_size new hypotheses (using normalized probabilities) and make them the current hypotheses.
            new_hypotheses = sorted(new_hypotheses, key=lambda x: x['pr'] / len(x['value']), reverse=True)
            current_hypotheses = new_hypotheses[:beam_size]

            i += 1

        # finally get the n best hypotheses from our current hypotheses 
        current_hypotheses = sorted(current_hypotheses, key=lambda x: x['pr'], reverse=True)
        n_best_hypotheses = current_hypotheses[:n_best]
        target_sentences = [hyp['value'] for hyp in n_best_hypotheses]

        # remove end of sequence symbol if existing
        for i in range(len(target_sentences)):
            if target_sentences[i][-1] == '</s>':
                target_sentences[i] = target_sentences[i][:-1]

    return target_sentences


'''
HELP FUNCTIONS
'''


# adds start of sequence symbols to a list of sentences depending on the window size
def add_start_of_sequence(sentences, dict_source, dict_target, window_size):
    start_of_sequence_index = dict_source.get_index_by_word('<s>')
    start_of_sequence = [start_of_sequence_index] * window_size
    for i in range(len(sentences)):
        sentences[i] = start_of_sequence + sentences[i]
    return sentences


# bleu for list of translations and unopened reference
def bleu_for_search(pred_target_sentences_list, reference_name):
    print('calculating the BLEU score ...')

    # Open and extract info from reference file
    path = os.path.join(os.path.dirname(sys.path[0]), 'test_data', reference_name)
    reference = open(path, 'r', encoding="utf-8")
    reference_text = reference.read().split('\n')
    reference.close()
    references = []
    for i in range(0, len(reference_text) - 1):
        references.append(reference_text[i].split())

    # Choose the best translation each as the hypothesis
    best_hypotheses = []
    for sentence_list in pred_target_sentences_list:
        best_hypotheses.append(sentence_list[0])

    bleu_score = bleu(best_hypotheses, references)
    print(' -> done.')
    print(f'\nbleu score: {bleu_score}\n')
    return bleu_score


# remove bpe
def remove_bpe(source_sentences, pred_target_sentences_list):
    print('removing bpe from source and target sentences ...')
    source_sentences = bpe.remove_bpe_from_sentence_list(source_sentences)
    pred_target_sentences_list = bpe.remove_bpe_from_sentences_list(pred_target_sentences_list)
    print(' -> done.')
    return source_sentences, pred_target_sentences_list


# SAVING
# this save method saves an overview for an easy review of how well search works, saving the n best translations as well as the source sentences and some more info
def save_overview(source_sentences, pred_target_sentences_list, source_name, reference_name, search_mode, beam_size,
                  n_best):
    output_path = os.path.join(os.path.dirname(__file__), 'search_output', reference_name + '[' + search_mode + ']')
    if search_mode == 'beam': output_path += f"[{beam_size}]"
    print(f'now writing an overview of the result of the search to {output_path} ...')
    with open(output_path, 'w', encoding="utf-8") as out_file:
        out_file.write(f"Translations for {source_name}, for each the best {n_best}" + '\n')
        for source_sentence, pred_target_sentences in zip(source_sentences, pred_target_sentences_list):
            out_file.write(f"\nPredicted translations for '{' '.join(source_sentence)}':\n")
            for i, pred_target_sentence in enumerate(pred_target_sentences):
                out_file.write(f"{i + 1}. '{' '.join(pred_target_sentence)}'\n")
    print(' -> done.')


# this save method saves the best translation to a file, sentence by sentence, similar to the test files multi30k
def save_best_translation(pred_target_sentences_list, reference_name, search_mode, beam_size):
    output_path = os.path.join(os.path.dirname(__file__), 'pred_data', reference_name + '[' + search_mode + ']')
    if search_mode == 'beam': output_path += f"[{beam_size}]"
    print(f'now writing best translation to {output_path} ...')
    with open(output_path, 'w', encoding="utf-8") as out_file:
        for pred_target_sentences in pred_target_sentences_list:
            out_file.write(f"{' '.join(pred_target_sentences[0])}\n")
    print(' -> done.')


'''
SEARCH
'''

'''
MAIN METHOD
'''
if __name__ == '__main__':

    # get config file as command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file_path', help='(relative) path of the config file')
    args = parser.parse_args()

    # open config file and get parameters
    f = open(args.config_file_path)
    config_data = json.load(f)

    # load the dictionaries
    dict_source = dictionary.Dictionary(config_data['source_dict_name'])
    dict_target = dictionary.Dictionary(config_data['target_dict_name'])

    # load the model from file
    # model = nnet.load_model(config_data['model_name'])['model'].to(nnet.device)
    model = nnet.load_model(config_data["model_name"])["model"]
    model.eval()

    # apply bpe to the source file and get indices (that also filter it through the vocabulary)
    source_sentences = bpe.get_text_with_applied_bpe(config_data['source_name'],
                                                     config_data['source_merge_operations_name'])
    source_sentences_indices = dict_source.from_text_to_indices(source_sentences)
    source_sentences_indices = add_start_of_sequence(source_sentences_indices, dict_source, dict_target,
                                                     model.window_size)

    # initialize a list of lists of the predicated target sentences (each a list of words (subwords)).
    # e.g. [
    # [["The","Weather","is","nice"],["Weather","is","good"]],
    # [["Machine","Learn","ing","is","cool"],["Learn","ing","machines","cool"]]
    # ] for a desired output of the best 2 translations
    pred_target_sentences_list = []

    # perform search using the model and the search datasets
    print(f"translating the source file ({config_data['search_mode']} mode) ...")
    # iterate over the source sentences
    for i, source_sentence_indices in enumerate(source_sentences_indices):
        if config_data['search_mode'] == 'greedy':
            pred_target_sentences = greedy_search(model=model, dict_source=dict_source, dict_target=dict_target,
                                                  source_sentence_indices=source_sentence_indices)

        elif config_data['search_mode'] == 'beam':
            pred_target_sentences = beam_search(model=model, dict_source=dict_source, dict_target=dict_target,
                                                source_sentence_indices=source_sentence_indices,
                                                beam_size=config_data['beam_size'], n_best=config_data['n_best'])

        pred_target_sentences_list.append(pred_target_sentences)

        if (i + 1) % 100 == 0:
            print(f' -> done for the first {i + 1} sentences.')

    print(' -> done.')

    # remove bpe if desired
    if (config_data['remove_bpe']):
        source_sentences, pred_target_sentences_list = remove_bpe(source_sentences, pred_target_sentences_list)

    # if target reference is given, we also calculate bleu
    if config_data['reference_name']:
        bleu_score = bleu_for_search(pred_target_sentences_list, config_data['reference_name'])

    # saving
    save_overview(source_sentences=source_sentences, pred_target_sentences_list=pred_target_sentences_list,
                  source_name=config_data['source_name'], reference_name=config_data['reference_name'],
                  search_mode=config_data['search_mode'], beam_size=config_data['beam_size'],
                  n_best=config_data['n_best'])
    save_best_translation(pred_target_sentences_list=pred_target_sentences_list,
                          reference_name=config_data['reference_name'], search_mode=config_data['search_mode'],
                          beam_size=config_data['beam_size'])
