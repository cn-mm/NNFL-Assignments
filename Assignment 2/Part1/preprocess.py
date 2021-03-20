from typing import List, Dict
import unidecode
import string
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import configparser
config = configparser.ConfigParser()
config.read("dev.config")
config=config["values"]


#Defining the global Tokens.
SOS_TOKEN = 1
EOS_TOKEN = 2
PAD_TOKEN = 0
MAX_LENGTH=20

# This function is preprocessing a single sentence from the database
#(1 mark)
def preprocess(sentence: str, hindi=False) -> str:
    # remove tabs and newlines
    sentence = ' '.join(sentence.split())
    ## Write your code below
    #convert the sentence into lower cases
    sentence = #CODE_BLANK_1. #Remove this comment before writing code here.
    # remove accented chars such as in cafe
    sentence = unidecode.unidecode(sentence)
    # remove punctuation
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    # remove digits
    sentence = sentence.translate(str.maketrans('', '', string.digits))
    # remove hindi digits
    if hindi:
        sentence = re.sub("[२३०८१५७९४६]", "", sentence)
    ##Write your code below
    #remove trailing and leading extra white spaces
    sentence = #CODE_BLANK_2
    ##Write your code below
    #Remove any excess white spaces from within the sentence
    sentence = #CODE_BLANK_3
    ##Write your code below
    #append the prepend the SOS token and append the EOS token to the sentence with spaces.
    sentence = #CODE_BLANK_4

    ''' for an expected input of sentence = ' hi, my name is  sam  ', output will be 'hi my name is sam'
    '''

    return sentence

#helper function 1. Returns a list of all the unique words in our corpora.
#(0.5 marks)
def get_vocab(lang: pd.Series) -> List:
    #Write your code here and remove the next line which says pass before you submit
    '''
       For a pd.series object like 
       1. Would you like some tea
       2. do you know my name

       output should be 
       ['do', 'know', 'like', 'my', 'name', 'some', 'tea', 'would', 'you']
    '''
    pass

#(0.5 marks)
#Helper 2: Creates a dictionary with token-> index mapping. Used in encoding.
def token_idx(words: List) -> Dict:
    #Write your code here and remove the next line which says pass before you submit
    '''
        input of words ['a','b','c'] -> output should be {'a': 1, 'b' : 2, 'c': 3}
    '''
    pass

#(0.5 marks)
#Helper 3: Creates a dictionary for index to word mapping. Used in decoding
def idx_token(wor2idx: Dict) -> Dict:
    #Write your code here and remove the next line which says pass before you submit
    pass
 

#Helper 4: Pads sequences to a particular length so that all the sequences are of same length in a batch.
def pad_sequences(sent_tensor: List[List[int]], max_len: int=20) -> np.ndarray:
    padded = np.zeros((max_len), dtype=np.int64)
    if len(sent_tensor) > max_len:
        padded[:] = sent_tensor[:max_len]
    else:
        padded[:len(sent_tensor)] = sent_tensor
    return padded

#Converts a particular sentence to its corresponding numeric tensor using word2idx dictionary.
def convert_to_tensor(word2idx: Dict, sentences: pd.Series):
    tensor = [[word2idx[s] for s in eng.split()]
              for eng in sentences.values.tolist()]
    tensor = [pad_sequences(x) for x in tensor]
    return tensor

#Class of type Dataset. This must contain __len__() and __getitem() functions as a part of their hooks.
class Data(Dataset):
    def __init__(self, input_sent, target_sent):
        self.input_sent = input_sent
        self.target_sent = target_sent

    def __len__(self):
        return len(self.input_sent)

    def __getitem__(self, index):
        x = self.input_sent[index]
        y = self.target_sent[index]
        return x, y

 #(1 mark)
#Main function being called when we need to retrieve inout batch, output batch and DataLoader objects.
def get_dataset(batch_size=2, types="train", shuffle=True, num_workers=1, pin_memory=False, drop_last=True):
    #Read the file 
    lines = pd.read_csv('Hindi_English_Truncated_Corpus.csv', encoding='utf-8')
    #Get lines only with source as "ted"
    lines = lines[lines['source'] == 'ted']  # Remove other sources
    ## Write your code below
    #Remove duplicate lines
    #CODE_BLANK_1
    
    #Random Sample of 25000 sentences
    lines = lines.sample(n=int(config["samples"]), random_state=42)
    ##Write your code below
    #Call preprocess functions on all english sentences
    #CODE_BLANK_2

    #Call preprocess functions on all hindi sentences
    #CODE_BLANK_3

    #Retrieve length of each english sentence and store it in the lines dataframe under a new column "length_english_sentence"
    #CODE_BLANK_4

    #Retrieve length of each hindi sentences and store it in the lines dataframe under a new column "length_hindi_sentence"
    #CODE_BLANK_5

    #Remove all the sentences with length less than max_length
    #CODE_BLANK_6
    #CODE_BLANK_7

    #Get List of english words
    # CODE_BLANK_8      
    
    #Get List of Hindi Words
    #CODE_BLANK_9

    #Get word2idx_eng for english
    #CODE_BLANK_10
    
    #Get word2idx_hin for hindi
    #CODE_BLANK_11

    #get idx2word_eng for english
    #CODE_BLANK_12

    #get idx2word_hin for hindi
    #CODE_BLANK_13
    
    #Convert the sentences to tensors using dictionaries created above
    #English tensor in input_tensor
#     CODE_BLANK_14
    #Hindi tensor in output_tensor
    #CODE_BLANK_15

    #Calling the split function and doing a 80-20 split on the input and target tensors.
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
        input_tensor, target_tensor, test_size=0.2, random_state=42)

    #Calling the Dataloaders and passing the Dataset type objects created using Data() class.
    if types == "train":
        train_dataset = Data(input_tensor_train, target_tensor_train)
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last), english_words, hindi_words
    elif types == "val":
        val_dataset = Data(input_tensor_val, target_tensor_val)
        return DataLoader(val_dataset, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last), idx2word_eng, idx2word_hin
    else:
        raise ValueError("types must be in ['train','val']")


if __name__ == "__main__":
    get_dataset()
