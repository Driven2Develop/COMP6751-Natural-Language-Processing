from contextlib import redirect_stdout
import sys
import pandas as pd
import pint
import re
import os
import time
from datetime import datetime

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
from gazzeteer import Gazetteer
from entityDetector import namedEntityDetector
from entityDetector import measuredEntityDetector
from grammar import contextFree

# Class definition, has an instance of each of the other modules
class PipelineDriver:
    def __init__(self, validationMode=False, verboseMode=False):
        self.validationMode = validationMode
        self.coNLL_headers = ['Word','POS_tag','Named_Entity','Entity','Measured_Entity']
        self.fileID = None
        self.documents = []
        self.validationDirectory = "validation_Text"
        self.validationPath = os.path.join(os.path.dirname(__file__), self.validationDirectory)
        self.gazetteer = Gazetteer()
        self.named_entity = namedEntityDetector()
        self.measured_entity = measuredEntityDetector()
        self.grammar = contextFree()
    
    # when the user selects option (i) they are prompted for a sentence to analyze, can be done repeatedly
    def get_sentence_from_user(self):
        response = input("Provide a sentence to analyze >")
        shortened_timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        filename = os.path.join(os.path.dirname(__file__), "output/cmdLine/{}.txt".format(shortened_timestamp))
        
        if not response.endswith('.'):
            response = response + '.'

        # output to command line and option to save locally
        output = self.process_sentence(response)
        print("Processed sentence >> {}".format(response))
        print("Printing POS tags >> {}".format(pos_tag(output[0])))
        driver.grammar.display(response) # show tree structure

        # output option
        user_input = input("Would you like to save the output to a file? (y/n)\n")
        while True:
            if user_input.lower().startswith('y'):
                with open(filename, 'w') as file:
                    file.write(str(pos_tag(output[0])))
                    with redirect_stdout(file):
                        driver.grammar.display(response)
                break
            elif user_input.lower().startswith('n'):
                break
            else:
                user_input = input("Invalid Entry. (y/n)")
        
        # repeat again
        user_input = input("Would you like to input another sentence? (y/n)\n")
        while True:
            if user_input.lower().startswith('y'):
                self.get_sentence_from_user() 
                break
            elif user_input.lower().startswith('n'):
                print("Exiting Application")
                exit()
            else:
                user_input = input("Invalid Entry. (y/n)")
    
    # reuters file ID option 
    def get_fileID_from_user(self, fileID):

        # Iterate through each category and get the file IDs to verify validitiy
        all_file_ids = []
        for category in reuters.categories():
            file_ids_in_category = reuters.fileids(category)
            all_file_ids.extend(file_ids_in_category)
        
        while True:
            if fileID in all_file_ids:
                self.documents.append(reuters.raw(fileID))
                self.fileID = fileID
                break
            else:
                fileID = input("Invalid file ID. Provide a valid file ID available in reuters corpus.\n")
    
    # processes an individual sentence with each feature. 
    def process_sentence(self, sentence):
        words = word_tokenize(sentence)
        pos_words = pos_tag(words)
        named_ent = driver.named_entity.get_ne(pos_words, False)
        gaz_ent = [driver.gazetteer.annotate_gazetteer(word) for word in words]
        measured_ent = driver.measured_entity.detect_with_pattern(sentence)

        # matches the headers within self.coNLL_headers
        return [words, [pos[1] for pos in pos_tag(words)], named_ent, gaz_ent, measured_ent]

    # if a document is obtained, process them one sentence at a time, saves individual sentence processing for future output
    def process_document(self, document):
        if driver.validationMode:
            print("Processing document with name {}".format(os.path.basename(document)))
            with open (document, 'r') as doc:
                document = doc.read()
        else:
            print("Processing document with ID {}".format(driver.fileID))

        output = {header: [] for header in driver.coNLL_headers}

        for sentence in sent_tokenize(document):
            processed_sentence = driver.process_sentence(sentence)

            for i, (key, value) in enumerate(output.items()):
                output[key].extend(processed_sentence[i])
        
        return output
    
    # helper function to get all documents within Validation_text directory. Runs analysis on these files in bulk
    def get_validation_files(self):
        file_paths = []
        for root, dirs, files in os.walk(self.validationPath):
            for file in files:
                file_paths.append(os.path.join(root, file))
        
        self.documents = file_paths

# initialization
print("COMP-6751 Project 2 Fall 2023. Iymen Abdella 40218280\n")
response = input("Would you like to run the validator? (v) Or provide a file ID for the Reuters corpus > Or input a sentence on command line? (i)")

# option choice
if response.lower() == 'v' or response.lower().startswith('v'):
    driver = PipelineDriver(validationMode=True)
    driver.get_validation_files()
elif response.lower() == 'i' or response.lower().startswith('i'):
    driver = PipelineDriver(validationMode=False)
    driver.get_sentence_from_user()
else:
    driver = PipelineDriver(validationMode=False)
    driver.get_fileID_from_user(response)

# main program loop
for doc in driver.documents:

    # saves document output for display and saving
    df = pd.DataFrame(driver.process_document(doc))
    shortened_timestamp = datetime.now().strftime("%m-%d_%H-%M")

    #file name based on type
    if driver.validationMode:
        tsv_filename = os.path.join(os.path.dirname(__file__), "output/{}_{}.tsv".format(os.path.splitext(os.path.basename(doc))[0], shortened_timestamp))
    else:
        tsv_filename = os.path.join(os.path.dirname(__file__), "output/{}_{}.tsv".format(driver.fileID, shortened_timestamp))

    df.to_csv(tsv_filename, sep='\t', index=True)
    print("Document output saved locally.")
    
    # option to show on command line
    response = input("Would you like to display the output in the command line? (y/n)\n")
    while True:
        if response.lower().startswith('y'):
            print(df)
            break
        elif response.lower().startswith('n'):
            break
        else:
            response = input("Invalid Entry. (y/n)")

    if (doc == driver.documents[-1] and driver.validationMode):
        break

    # REMOVE
    if driver.validationMode:
        # with open (doc, 'r') as d:
        #     document = d.read()
        # for sentence in sent_tokenize(document):
        #     driver.grammar.display(sentence)

        response = input("Would you like to process the next document? (y/n)\n")

        #process next document or exit   
        while True:
            if response.lower().startswith('y'):
                print("Processing next document.")
                break
            elif response.lower().startswith('n'):
                print('Exiting application.')
                exit()
            else:
                response = input("Invalid Entry. (y/n)")
    else:
        # search for another document or exit
        response = input("Would you like to process another reuters corpus document? (y/n)\n")
        while True:
            if response.lower().startswith('y'):
                fileID = input("Provide a file ID for the Reuters corpus >")
                driver.get_fileID_from_user(fileID)
                break
            elif response.lower().startswith('n'):
                print('Exiting application.')
                exit()
            else:
                response = input("Invalid Entry. (y/n)")

print("All Documents processed. Exiting Application.")