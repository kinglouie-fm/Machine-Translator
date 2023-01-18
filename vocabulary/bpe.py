import argparse
import sys
import gzip
import os.path
import pickle

"""
HELP FUNCTIONS NEEDED FOR LEARNING THE MERGE OPERATIONS AND APPLYING THEM ON TEXT
"""


# help function that opens a text file and converts the content to list of sentences (which here are lists of words)
# text_file_name: name (/path) of the text file
def get_sentence_list_from(text_file_name):
    # open the text file and save its content in text_content
    path = os.path.join(os.path.dirname(sys.path[0]), 'test_data', text_file_name)
    text_file = gzip.open(path, 'rt', encoding='utf-8')
    print(f"reading from file '{path}'...")
    try:
        text_content = text_file.readlines()
    except OSError:
        text_file.close()
        text_file = open(path, 'r', encoding='utf-8')
        text_content = text_file.readlines()
    text_file.close()
    print(f"done reading from file '{path}'...")

    # fill the lists sentences_as_lists by converting the sentences to lists of words
    sentences_as_lists = [sentence.split() for sentence in text_content]
    return sentences_as_lists


# return dictionary with word frequencies
# sentences_as_lists: list of sentences (as lists of words)
def word_freq(sentences_as_lists):
    word_freq_dict = {}
    for sentence in sentences_as_lists:
        for word in sentence:
            if word_freq_dict.get(word):
                word_freq_dict[word] = word_freq_dict.get(word) + 1
                # word_freq_dict.update({word: word_freq_dict.get(word) + 1})
            else:
                word_freq_dict[word] = 1
                # word_freq_dict.update({word: 1})
    return word_freq_dict


# takes a dictionary and adds @@-separations between all the keys
# old_word_freq_dict: dictionary with word frequencies / output of func word_freq 
def subword_init(old_word_freq_dict):
    new_word_freq_dict = {}
    for word in old_word_freq_dict.keys():
        new_word_freq_dict[' '.join(word) + '</w>'] = old_word_freq_dict[word]
    return new_word_freq_dict


'''
FUNCTIONS FOR LEARNING AND SAVING THE MERGE OPERATIONS
'''


# function that updates our word freq dict by merging two subwords together each time the function is called
# word_freq_dict is a dictionary with separations (@@)
def learn_one_merge_operation(word_freq_dict):
    """
    DETERMINE MOST FREQUENT PAIR
    """
    pairs_merged_dict = {}  # pairs_merged_dict: {"in": ("i","n"), ...}
    pair_dict = {}  # pair_dict saves every pair with its occurrences

    for word in word_freq_dict.keys():
        subwords = list(filter(None, word.split(' ')))
        # w = "@@o@@k@@" -> ["o","k"]
        for i in range(0, len(subwords) - 1):
            # if pair exists then frequency + number of occurrences of the word
            if pair_dict.get(subwords[i] + subwords[i + 1]):
                pair_dict[subwords[i] + subwords[i + 1]] = pair_dict.get(
                    subwords[i] + subwords[i + 1]) + word_freq_dict.get(word)
            # else frequency set to number of occurrences of the word and merge operations saved in tuple
            else:
                pair_dict[subwords[i] + subwords[i + 1]] = word_freq_dict.get(word)
                pairs_merged_dict[subwords[i] + subwords[i + 1]] = (subwords[i], subwords[i + 1])

    if not pair_dict and not pairs_merged_dict:
        print("PAIR NOT IN MERGED PAIR DICT")
        return None, None

    # get most freq pair (tuple of two subwords) and save in most_freq_pair
    most_freq_pair = pairs_merged_dict.get(max(pair_dict, key=pair_dict.get))

    """
    MERGING BY REMOVING THE @@s
    """
    # create and fill a new word freq dict in which the keys have merged the two subwords together
    new_word_freq_dict = {}
    for word in word_freq_dict.keys():
        subwords = word.split(' ')
        i = 0
        while i < (len(subwords) - 1):
            if subwords[i] == most_freq_pair[0] and subwords[i + 1] == most_freq_pair[1]:
                subwords[i: i + 2] = [''.join(subwords[i: i + 2])]
            i += 1
        word_after_merge = ' '.join(subwords)
        new_word_freq_dict[word_after_merge] = word_freq_dict[word]

    return new_word_freq_dict, most_freq_pair


# return list with tuples based on n merge operations
# text_file_names: names of the text files we want to learn the operations on (list)
# n: the number of merge operations
def learn_merge_operations(text_file_names, n):
    # open text files and derive sentences from them
    sentences_as_lists = []
    for text_file_name in text_file_names:
        sentences_as_lists.extend(get_sentence_list_from(text_file_name))

    word_freq_dict = subword_init(word_freq(sentences_as_lists))  # dict with the frequency of words, separated by @@s
    merge_operations = []  # list of tuples

    # call apply_one_subword_operation n times
    for i in range(0, n):
        if i % 100 == 0 and i != 0:
            print(f"applied first {i} merge operations")
        new_word_freq_dict, most_freq_pair = learn_one_merge_operation(word_freq_dict)
        if most_freq_pair:
            merge_operations.append(most_freq_pair)
            word_freq_dict = new_word_freq_dict
        else:
            break  # no most frequent pair was determined, so we do not need to call our function again

    return merge_operations


# creates a text file {n}_operations_{text_file_name} with a list of subword operations
# text_file_names: names of the text files we want to learn the operations on (list)
# n: number of merge operations
def learn_and_save_merge_operations(text_file_names, n):
    print(f"starting to apply {n} merge operations...")

    # determine (learn) the merge operations
    merge_operations = learn_merge_operations(text_file_names, n)

    # determine the right file name and save the merge operations to the file
    path = os.path.join(os.path.dirname(__file__), 'merge_operations', f'{n}_operations_{text_file_names}')
    print(f"finished getting merge operations, now writing them to file {path}...")
    with open(path, 'w', encoding="utf-8") as output_file:
        for merge_operation in merge_operations:
            output_file.write(f"{merge_operation[0]},{merge_operation[1]}\n")
    print("Done writing operations to file")


"""
FUNCTIONS FOR APPLYING THE LEARNED MERGE OPERATIONS ON TEXT
"""


# return value: a dictionary with old words as keys and the words applied with bpe as values
# text_file_name: name (/path) of the text file
# merge_operations_file_name: name of the file name containing the merge operations
# force_gen: forces generation of a new .bpe file and removes the old, no values are taken from the old file.
def apply_bpe(text_file_name, merge_operations_file_name, force_gen=False):
    # holds the original word as key and the word with the applied operations as value
    old_new_dict = {}

    # file_path where the file (text_file_name with applied bpe using merge_operations_file_name for applying the merge operations).
    # should be found or where it should be saved to if it does not exist
    bpe_path = os.path.join(os.path.dirname(__file__), 'bpe_dicts',
                            text_file_name + "_" + merge_operations_file_name + ".pkl")

    if os.path.exists(bpe_path) and not force_gen:  # File exists and shall not be overwritten, thus read it
        print(f"file {bpe_path} already exists, reading from file...")

        old_new_dict = pickle.load(open(bpe_path, "rb"))  # Loads the saved file into old_new_dict

    else:  # File does not exist, create it
        print(f"file {bpe_path} does not exist, creating and saving the applied bpe ...")

        # open the merge operations file and save the content in merge_operations
        merge_path = os.path.join(os.path.dirname(__file__), 'merge_operations', merge_operations_file_name)
        f = open(merge_path, 'rt', encoding="utf-8")
        merge_operations = list(filter(None, f.read().split('\n')))  # filter removes the empty strings at start and end
        f.close()

        # initialize the word freq dictionary
        sentences_as_lists = get_sentence_list_from(text_file_name)
        word_freq_dict = subword_init(word_freq(sentences_as_lists))

        print("Determining the bpe for the given words one by one...")
        # print a progress update every 1000 words
        for i, word in enumerate(word_freq_dict.keys()):
            if i % 1000 == 0 and i != 0:
                print(f"Done for the first {i} words")

            subwords = word.split(' ')

            # iterate over every merge operation and determine whether we can apply it to our word
            # (i.e. pull two subwords together)
            # -> if we can apply a subword operation (if condition) we pull together the list entries
            # of two subwords
            for merge_operation in merge_operations:
                pair = merge_operation.split(",")
                j = 0
                # Checks if the current merge_op is applicable to the current subword and the one after, applies if so
                while j < (len(subwords) - 1):
                    if subwords[j] == pair[0] and subwords[j + 1] == pair[1]:
                        subwords[j: j + 2] = [''.join(subwords[j: j + 2])]
                    j += 1

            # create a dictionary with entries for every original word:
            # - key is the original word (string)
            # - value is the list of subwords
            original_word = ''.join(subwords)  # Works, but using the current word from word_freq_dict should be easier
            old_new_dict[original_word] = subwords

        pickle.dump(old_new_dict, open(bpe_path, "wb"))

    # print(old_new_dict)
    return old_new_dict


# returns the initial text with applied operations as list of list: WE DID THIS TO GET ORDER
# text_file_name: name (/path) of the text file we want to apply bpe on
# merge_operations_file_name: name of the file name containing the merge operations
def get_text_with_applied_bpe(text_file_name, merge_operations_file_name, force_gen=False):
    # get bpe dictionary
    old_new_dict = apply_bpe(text_file_name, merge_operations_file_name, force_gen)
    # apply the dictionary substitutions to our text
    sentences_as_lists = get_sentence_list_from(text_file_name)
    print("applying BPE to our text ...")
    text_with_applied_bpe = []  # list of sentences (one sentence = one string)
    for sentence in sentences_as_lists:
        # print(sentence)
        sentence_with_applied_bpe = []
        for word in sentence:
            word += '</w>'  # bpe has the </w> which the original text hasn't
            # print(old_new_dict.get(word))
            sentence_with_applied_bpe.extend(old_new_dict.get(word))
        text_with_applied_bpe.append(sentence_with_applied_bpe)
    print(" -> done.")
    return text_with_applied_bpe


# return an alphabetically ordered list of all subwords that applying bpe on our sentences file yields
# text_file_name: name (/path) of the text file we want to apply bpe on
# merge_operations_file_name: name of the file name containing the merge operations
def get_subwords(text_file_name, merge_operations_file_name, force_gen=False):
    print("getting a list of subwords bpe yields on the given text...")
    look_up_dict = apply_bpe(text_file_name, merge_operations_file_name, force_gen)

    # create empty set of subwords and fill it using the for loop
    subword_set = set()
    for subwords in look_up_dict.values():
        print(subwords)
        subword_set.update(set(subwords))

    # create an alphabetically ordered list of subwords from the set and return it
    ordered_subword_list = sorted(list(subword_set))
    return ordered_subword_list


# def get_subwords(bpe_dict_file_name):
#     bpe_path = os.path.join(os.path.dirname(__file__), 'bpe_dicts', bpe_dict_file_name)
#     bpe_dict = pickle.load(open(bpe_path, "rb"))

#     # create empty set of subwords and fill it using the for loop
#     subword_set = set()
#     for split_word in look_up_dict.values():
#         subwords = filter(None, split_word.split("@@ "))  # filter removes the empty strings at the start and the end
#         subword_set.update(subwords)

#     # create an alphabetically ordered list of subwords from the set and return it
#     ordered_subword_list = sorted(list(subword_set))
#     return ordered_subword_list


# removes the bpe from a text, thus merging list entries of subwords that would belong together without bpe
# text: list of lists of sentences (each a list of subwords)
# TODO finish implementation
def remove_bpe_from_sentence_list(sentences):

    new_sentences = []
    for sentence in sentences:
        new_sentence = []
        new_word = ""
        for subword in sentence:
            # print(subword)
            new_word += subword
            if subword.endswith('</w>'):
                new_word = new_word[:-4]  # Removes the '</w>' at the end of a word. more efficient then .replace(...)
                new_sentence.append(new_word)
                new_word = ""
        new_sentences.append(new_sentence)
    return new_sentences

def remove_bpe_from_sentences_list(text):

    new_text = []
    for sentences in text:
        new_text.append(remove_bpe_from_sentence_list(sentences))
    return new_text



"""
TESTING
"""
if __name__ == '__main__':
    """
    MERGE TEST
    """
    TEXT_FILE_NAMES = ['multi30k.en.gz', 'multi30k.de.gz']
    NUMBER_OF_OPERATIONS = 7000
    # learn_and_save_merge_operations(TEXT_FILE_NAMES, NUMBER_OF_OPERATIONS)

    """
    APPLY TEST
    """
    TEXT_FILE_NAME = 'multi30k.en.gz'
    MERGE_OPERATIONS_FILE_NAME = "7000_operations_['multi30k.en.gz', 'multi30k.de.gz']"
    # print(get_text_with_applied_bpe(TEXT_FILE_NAME, MERGE_OPERATIONS_FILE_NAME))
    print(remove_bpe(get_text_with_applied_bpe(TEXT_FILE_NAME, MERGE_OPERATIONS_FILE_NAME)))
    # print(get_subwords(TEXT_FILE_NAME, MERGE_OPERATIONS_FILE_NAME))
