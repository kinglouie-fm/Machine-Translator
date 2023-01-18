import gzip
import os.path
import dictionary
import bpe


# alignment function for domain '1 to max_domain', range '1 to max_range' applied to value i (defined for value up to max_range -1)
# max_domain equals max_target, max_range equals max_source
def alignment(max_domain, max_range, i):
    return round((max_range/max_domain) * i)


# corresponding dict: the dictionary to look up the word
# sentence: list of words (in either target or source)
# index: note that index 1 should give us the first item in the list
# this function first determines whether the given index is inside the scope 1 to len(sentence) to determine whether we need to insert a corresponding word or a sequence symbol. Then it asks the dictionary to give the corresponding number (index) for the word in the dictionary
def get_index_for_index(corresponding_dict, sentence, index):

    # determine right string
    if index <= 0:
        word = "<s>"
    elif index <= len(sentence):
        word = sentence[index-1]
    else:
        word = "</s>"

    return corresponding_dict.get_index_by_word(word)


# This function further prepares given data for training/development. It replaces strings by indizes and groups data into lists S, T and L according to a given window size
# source_sentences: list of source sentences, bpe applied and filtered through vocabulary
# target_sentences: list of target sentences, bpe applied and filtered through vocabulary
def training_preparation(source_sentences, target_sentences, dict_source, dict_target, window_size):

    print("Preparing our data for further processing in the model by creating S, T and L and looking up indices for our words...")
    lines = []  # list of lines

    # repeat for every sentence
    for (source_sentence, target_sentence) in zip(source_sentences, target_sentences):

        J = len(source_sentence)
        max_source = J+1 # needed for alignment function (range)
        I = len(target_sentence)
        max_target = I+1 # needed for alignment function (domain) and iteration on matrix

        """
        CREATE DATA LINE BY LINE
        """
        # we now create or matrizes line by line iterating over the target words for 1 to max_target
        # each iteration of this loop creates one row
        for target_position in range(1, max_target+1):

            source_position = alignment(max_target, max_source, target_position)

            # line in matrix s
            line_S = []
            for position in range(source_position - window_size, source_position + window_size + 1):
                line_S.append(get_index_for_index(dict_source, source_sentence, position))

            # line in matrix t
            line_T = []
            for position in range(target_position - window_size, target_position):
                line_T.append(get_index_for_index(dict_target, target_sentence, position))

            # line in matrix l
            position = target_position
            line_L = [get_index_for_index(dict_target, target_sentence, position)]

            # create and fill dictionary for the line of the batch
            line = [line_S, line_T, line_L]
            # add line to our current set of lines
            lines.append(line)

    return lines


# save a list of the lines to a text file
def save_training_preparation(data, dict_source, dict_target):
        
    print("Saving our prepared data...")
    """
    SAVE DATA (the indizes)
    """
    path = os.path.join(os.path.dirname(__file__), 'dataset_preparations', 'prepared_data.txt')
    with open(path, 'w', encoding = "utf-8") as out_file:
        for line in data:
            out_file.write(str(line) + '\n')

    """
    SAVE DATA (as strings)
    """
    # first replace the int values in the data by the corresponding words from the dictionary
    for line in data:
        for i in range(len(line[0])):
            line[0][i] = dict_source.get_word_by_index(line[0][i])
        for i in range(len(line[1])):
            line[1][i] = dict_target.get_word_by_index(line[1][i])
        line[2][0] = dict_target.get_word_by_index(line[2][0])

    path = os.path.join(os.path.dirname(__file__), 'dataset_preparations', 'prepared_data_as_strings.txt')
    with open(path, 'w', encoding = "utf-8") as out_file:
        for line in data:
            out_file.write(str(line) + '\n')



if __name__ == '__main__':
    # constants
    SOURCE_SENTENCES_FILE_NAME = 'multi30k.de.gz'
    TARGET_SENTENCES_FILE_NAME = 'multi30k.en.gz'
    SOURCE_MERGE_OPERATIONS_FILE_NAME = "7000_operations_['multi30k.en.gz', 'multi30k.de.gz']"
    TARGET_MERGE_OPERATIONS_FILE_NAME = "7000_operations_['multi30k.en.gz', 'multi30k.de.gz']"
    WINDOW_SIZE = 2

    # apply bpe on the files (by giving the file name)
    source_sentences = bpe.get_text_with_applied_bpe(SOURCE_SENTENCES_FILE_NAME, SOURCE_MERGE_OPERATIONS_FILE_NAME)
    target_sentences = bpe.get_text_with_applied_bpe(TARGET_SENTENCES_FILE_NAME, TARGET_MERGE_OPERATIONS_FILE_NAME)

    # create dictionary and filter through it
    dict_source = dictionary.Dictionary(SOURCE_SENTENCES_FILE_NAME, SOURCE_MERGE_OPERATIONS_FILE_NAME)
    dict_target = dictionary.Dictionary(TARGET_SENTENCES_FILE_NAME, TARGET_MERGE_OPERATIONS_FILE_NAME)
    source_sentences = dict_source.filter_text_by_vocabulary(source_sentences)
    target_sentences = dict_target.filter_text_by_vocabulary(target_sentences)

    # call the training preparation method and save the output to a file
    data = training_preparation(source_sentences, target_sentences, dict_source, dict_target, WINDOW_SIZE)
    save_training_preparation(data, dict_source, dict_target)
