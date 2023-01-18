from bidict import bidict
import bpe
import os.path


class Dictionary:
    # text_file_name: string (name of the file)
    # if len[args] = 1 we directly give the vocab file
    # if len[args] = 2 we give text_file_name and merge_operations_file_name and can create a new vocabulary if a corresponding does not exist
    def __init__(self, *args, force_gen=False):
        # initialize the vocabulary and add the symbols we always need
        self.vocab = bidict()
        self.vocab[0] = '<s>'
        self.vocab[1] = '</s>'
        self.vocab[2] = '<UNK>'
        
        # We are given the vocab file directly
        if len(args) == 1:
            path = os.path.join(os.path.dirname(__file__), 'vocabulary', args[0])

        # We are given a text file name, a merge operations file name
        elif len(args) == 2:
            text_file_name = args[0]
            merge_operations_file_name = args[1]

            # Reads a dictionary if it already exists, else creates the file for later use
            path = os.path.join(os.path.dirname(__file__), 'vocabulary', text_file_name + '.vocab')
            if os.path.exists(path):
                print(f"vocab file {path} already exists, reading from file...")
            else:
                print(f"vocab file {path} does not exist yet, contacting bpe for generation...")
                words = bpe.get_subwords(text_file_name, merge_operations_file_name)  # list of all sub-words in the text
                print("Done generating vocab file")
                with open(path, 'w', encoding="utf-8") as out_file:
                    for word in words:
                        out_file.write(word + "\n")

        # create the vocab from vocab file (that surely exists because we generated it in case it did not)
        with open(path, 'rt', encoding="utf-8") as vocab_file:
            try:
                for index, word in enumerate(vocab_file):
                    self.vocab[index+3] = word.replace("\n", "")
            except:
                print(f'Vocabulary file {path} does not exist, exiting...')

    def __len__(self):
        return len(self.vocab)

    def contains_word(self, word):
        return word in self.vocab.inverse

    def contains_index(self, index):
        return index in self.vocab

    def get_word_by_index(self, index):
        if self.contains_index(index):
            return self.vocab[index]
        else:
            return '<UNK>'

    def get_index_by_word(self, word):
        if self.contains_word(word):
            return self.vocab.inverse[word]
        else:
            return self.vocab.inverse['<UNK>']


    # text: list of sentences (each as a list of strings)
    def filter_text_by_vocabulary(self, text):
        print("filtering text by vocabulary...")
        output = []
        for sentence in text:
            for index, word in enumerate(sentence):
                if word not in self.vocab.inverse:
                    sentence[index] = '<UNK>'
            output.append(sentence)
        return output

    # text: list of sentences (each as a list of indices)
    def from_indices_to_text(self, text):
        output = []
        for sentence in text:
            output.append([self.get_word_by_index(index) for index in sentence])
        return output

    # text: list of sentences (each as a list of strings)
    def from_text_to_indices(self, text):
        output = []
        for sentence in text:
            output.append([self.get_index_by_word(word) for word in sentence])
        return output

# create dictionaries for the two languages and do some tests on them
if __name__ == '__main__':
    MERGE_OPERATIONS_FILE_NAME = "7000_operations_['multi30k.en.gz', 'multi30k.de.gz']"
    dict_en = Dictionary('multi30k.en.gz', MERGE_OPERATIONS_FILE_NAME)
    # print(dict_en.get_word_by_index(50))
    # print(dict_en) # TODO implement