import argparse
import json
import torch
import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'metrics'))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'vocabulary'))
from bleu import bleu
import ffmodel
import bpe
import search
import dictionary
import plot

'''
Calculate stats for all checkpoints (bleu, perplexity)
'''


# reference is a text file
def stats(model_name, source_name, reference_name, source_merge_operations_file_name, target_merge_operations_file_name,
          dict_source, dict_target, beam_size):
    # prepare source
    source_sentences = bpe.get_text_with_applied_bpe(source_name, source_merge_operations_file_name)
    source_sentences_indices = dict_source.from_text_to_indices(source_sentences)
    # prepare reference
    reference_sentences = bpe.get_text_with_applied_bpe(reference_name, target_merge_operations_file_name)

    # For every file/checkpoint in folder, add an entry to our list of dictionaries, lists
    stats = []
    for load_dict in ffmodel.load_models(model_name):

        model = load_dict['model'].to(ffmodel.device)
        model.eval()

        # further prepare source depending on the model window size
        source_sentences_indices = search.add_start_of_sequence(source_sentences_indices, dict_source, dict_target,
                                                                model.window_size)

        # target
        pred_target_sentences_list = []

        # perform search using the model and the search datasets
        print(f"translating the source file ...")
        for i, source_sentence_indices in enumerate(source_sentences_indices):
            pred_target_sentences = search.beam_search(model=model, dict_source=dict_source, dict_target=dict_target,
                                                       source_sentence_indices=source_sentence_indices,
                                                       beam_size=beam_size, n_best=1)
            pred_target_sentences_list.append(pred_target_sentences)

            if (i + 1) % 100 == 0:
                print(f' -> done for the first {i + 1} sentences.')

        print(' -> done.')

        # remove bpe again from source and target 
        print('removing bpe from source and target sentences ...')
        source_sentences = bpe.remove_bpe_from_sentence_list(source_sentences)
        pred_target_sentences_list = bpe.remove_bpe_from_sentences_list(pred_target_sentences_list)
        print(' -> done.')

        # calculate bleu
        print('calculating the BLEU score ...')
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

        # create stats dict and add it to the list
        stats_dict = {
            'name': load_dict['filename'],
            'bleu': bleu_score,
            'perp': load_dict['perplexity']
        }
        stats.append(stats_dict)

    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file_path", help="(relative) path of the config file")
    args = parser.parse_args()

    # open config file and get parameters
    f = open(args.config_file_path)
    config_data = json.load(f)
    f.close()

    # dictionaries
    dict_source = dictionary.Dictionary(config_data['source_dict_name'])
    dict_target = dictionary.Dictionary(config_data['target_dict_name'])

    # calculate stats for every checkpoint
    stats_list = stats(config_data['model_name'], config_data['source_name'], config_data['reference_name'],
                       config_data['source_merge_operations_name'], config_data['target_merge_operations_name'],
                       dict_source,
                       dict_target, config_data['beam_size'])
    plot.plot(stats_list)
