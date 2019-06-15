import os
import pickle
import re

class Vocabulary(object):
    """
    The class is used to form a vocabulary (bag-of-words)
    
    Attributes:
    -----------
    number_of_words
        current number of words in the class instance
        
    word_to_id
        dictionary mapping words (tokens) to their ids
        
    id_to_word
        dictionary mapping ids to the corresponding words
    -----------
    """
    
    def __init__(self):
        self.number_of_words = 1
        self.word_to_id = dict()
        self.id_to_word = dict()
        self.word_to_id['<un>'] = 0
        self.id_to_word[0] = '<un>'
    
    def add_word(self, word):
        """
        Adds a word in the vocabulary
        
        Parameters:
        -----------
        word : str
            The word to add
        -----------
        """
        if word not in self.word_to_id:
            self.word_to_id[word] = self.number_of_words
            self.id_to_word[self.number_of_words] = word
            self.number_of_words += 1
                
    def get_id_by_word(self, word):
        """
        Returns id for an input word
        
        Parameters:
        -----------
        word : str
            The word for which id is needed
        -----------
        """
        return self.word_to_id[word]
    
    def get_word_by_id(self, idx):
        """
        Returns a word for an input id
        
        Parameters:
        -----------
        id : int
            The id for which word is needed
        -----------
        """
        return self.id_to_word[idx]
    
    def save_vocabulary(self, filename_word_to_id='word_to_id.pickle', filename_id_to_word='id_to_word.pickle'):
        """
        Saves vocabulary dictionaries to pickle files
        
        Parameters:
        -----------
        filename_word_to_id : str
            The filename for word_to_id dictionary
        
        filename_id_to_word : str
            The filename for id_to_word dictionary
        -----------
        """
        try: 
            os.mkdir('./vocabulary')
        except:
            pass
        path_word_to_id = os.path.join('./vocabulary/', filename_word_to_id)
        path_id_to_word = os.path.join('./vocabulary/', filename_id_to_word)
        
        with open(path_word_to_id, 'wb') as writer:
            pickle.dump(self.word_to_id, writer)
            
        with open(path_id_to_word, 'wb') as writer:
            pickle.dump(self.id_to_word, writer)
            
    def load_vocabulary(self, path_word_to_id='./vocabulary/word_to_id.pickle', path_id_to_word='./vocabulary/id_to_word.pickle'):
        """
        Loads vocabulary dictionaries from pickle files
        
        Parameters:
        -----------
        path_word_to_id : str
            The path to file with word_to_id dictionary
        
        filename_id_to_word : str
            The path to file with id_to_word dictionary
        -----------
        """
        with open(path_word_to_id, 'rb') as reader:
            self.word_to_id = pickle.load(reader)
            
        with open(path_id_to_word, 'rb') as reader:
            self.id_to_word = pickle.load(reader)
            
        self.number_of_words = len(self.word_to_id)


def preprocess_captions(all_captions):
    """
    Replaces all the signs by whitespaces
    
    Parameters:
    -----------
    all_captions: list
        List of lists with all the captions
    -----------
    """
    for captions_list in all_captions:
        for i, caption in enumerate(captions_list):
            captions_list[i] = re.sub('\W+', ' ', caption.lower())


def add_start_and_end_to_captions(all_captions, start_str = '<SOS>', end_str = '<EOS>'):
    """
    Adds start and end of caption markers
    
    Parameters:
    -----------
    all_captions: list
        List of lists with all the captions
        
    start_str: str
        Start of caption marker
        
    end_str: str
        End of caption marker
    -----------
    """
    for captions in all_captions:
        for i in range(len(captions)):
            captions[i] =  '{} {} {}'.format(start_str, captions[i], end_str)
            captions[i] = captions[i].replace('  ', ' ').lower()
