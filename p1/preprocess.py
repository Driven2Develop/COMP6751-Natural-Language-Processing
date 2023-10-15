import sys
import pandas as pd
import pint
import re
import os
import time

import nltk
nltk.download('reuters') #corpus
nltk.download('punkt') #tokenizer
nltk.download('averaged_perceptron_tagger') #POS tagger
nltk.download('maxent_ne_chunker')
nltk.download('words')

from nltk.probability import FreqDist
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk import ne_chunk

from collections import OrderedDict

# get country and currency gazetteer info from geonames. Get units gazetteer from PINT
# convert all to lower case to facilitate  comparison
gaz_path = os.path.join(os.path.dirname(__file__), "gazetteers/countries.tsv")
gaz_country = pd.read_csv(gaz_path, sep='\t', header=0, usecols=['Country'])['Country'].tolist()
gaz_country = [s.lower() for s in gaz_country]

gaz_currency = pd.read_csv(gaz_path, sep='\t', header=0, usecols=['CurrencyName'])['CurrencyName'].tolist()
gaz_currency = [s.lower() for s in gaz_currency]

plural_units = ["tonnes"]
gaz_units = list(pint.UnitRegistry()._units.keys())
gaz_units = [s.lower() for s in gaz_units]
gaz_units.append(plural_units)

# helper method to compare with gazzetteer
def annotate_gazetteer(word, pos_word):
    word = word.lower()
    word_singular = word
    if word[-1] == 's': # edge case to handle units that end in 's' like meters or seconds
            word_singular = word[:-1]

    unnaccepted_pos = ['IN', 'DT', 'RB', 'JJ', 'VBZ']
    if (pos_word in unnaccepted_pos): #edge case when preposition misconstrued with unit of measurement
        return
    if word in gaz_country or word_singular in gaz_country:
        return 'Country'
    elif word in gaz_currency or word_singular in gaz_currency:
        return 'Currency'
    elif word in gaz_units or word_singular in gaz_units:  
        return 'Unit'

#handles measured entity annotation
def annotate_measured(sentence, matches):

    measured_ent = []
    words = word_tokenize(sentence)
    pos_tags = pos_tag(words)

    # if next word is a noun combine them, otherwise ignore.
    for i, word in enumerate(pos_tags):
        if word[0] in matches and pos_tags[i+1][1] in ['NN', 'NNS', 'NNP', 'CD'] : 
            measured_ent.append("{} {}".format(word[0], pos_tags[i+1][0]))

    return measured_ent

# helper method that returns a list of all the detected named entities from named entity tree
def extract_entities(tree):
    named_entities = []
    if hasattr(tree, 'label') and tree.label() == 'NE':
        named_entities.append(' '.join([child[0] for child in tree]))
    for subtree in tree:
        if type(subtree) == nltk.Tree:
            named_entities.extend(extract_entities(subtree))
    return named_entities

#read user input for file ID, default if none provided 
if len(sys.argv) > 1:
    fileID = sys.argv[1]
else:
    fileID = "training/267"

sent_headers = ['Word', 'POS_tag', 'Named_Entity', 'Gazetteer_Annotation']
row_list = []
measured_ent = []

for sentence in sent_tokenize(reuters.raw(fileID)):  

    words = word_tokenize(sentence)
    named_ent = extract_entities(ne_chunk(pos_tag(words), binary=True)) # list of ne in sentence

    # measured entity detection in sentence, rather than individual words 
    # Loop through each pattern and find all matches in the sentence
    patterns = [r'\d+', r'\d{1,3}(?:,\d{3})+', r'\d+\.\d+']
    matches = []

    for pattern in patterns:
        matches.extend(re.findall(pattern, sentence))

    if len(matches) < 1 : 
        continue
    else:
        measured_ent.extend(annotate_measured(sentence, matches))

    # loop for creating dataframe with POS, NE, and gazetteer annotation
    for word, pos_word in pos_tag(words):
        
        named_entity = 'n/a'
        if word in named_ent:
            named_entity = 'NE'
        
        row = {'Word': word, 
               'POS_tag': pos_word, 
               'Named_Entity': named_entity, 
               'Gazetteer_Annotation': annotate_gazetteer(word, pos_word)
               }
        
        row_list.append(row)

#save dataframe to an external csv file as well. 
csv_filename = os.path.join(os.path.dirname(__file__), "output/{}_{}.csv".format(fileID, time.time()))
df = pd.DataFrame(row_list, columns=sent_headers)
df.to_csv(csv_filename)
print("Measured entities {}".format(measured_ent))
print("Saved dataframe to csv file {}. Printing a sample of the dataframe below: ".format(csv_filename))
print(df)