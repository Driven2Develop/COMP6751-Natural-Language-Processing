import os
import nltk
nltk.download('reuters')
from nltk.corpus import reuters

from grammar import ContextFree, FeatureGrammar, AfinnBaseline
from nltk.tokenize import sent_tokenize
from preprocess import PipelineDriver

def get_pipeline_options():
    feature_grammar_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'grammars', 'feature_grammar.fcfg')
    baseline_grammar_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'grammars', 'AFINN-111.txt')
    response = input("Would you like to run the validator? (v) Or input a sentence on command line? (i) Or provide a file ID for the Reuters corpus >> ")
    
    grammar_response = input("Would you like to use the context free grammar? (c) or the feature and sentiment grammar? (f) ro the baseline afinn sentiment? (b) ")
    while True:

        if grammar_response.lower().startswith('c'):
            grammar = ContextFree()
            break
        elif grammar_response.lower().startswith('f'):
            grammar = FeatureGrammar(feature_grammar_path)
            break
        elif grammar_response.lower().startswith('b'):
            grammar = AfinnBaseline(baseline_grammar_path)
            break
        else:
            grammar_response = input ("Invalid response. Would you like to use the context free grammar? (c) or the feature and sentiment grammar? (f) ")
    
    # Mode of fetching sentences. 
    # Validation processes all the sentences from the files within the validation_Text directory
    # File ID allows user to provide a reuters corpus from which its sentences will be processed.
    if response.lower().startswith('v'):
        sentences = get_validation_files()

    elif response.lower().startswith('i'):
        sentences = get_sentence_from_user()
    else:
        sentences = get_fileID_from_user(response)

    return grammar, sentences

# takes all the files in the validation folder and adds them as sentences
def get_validation_files():
    validation_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'validation_Text')
    file_paths = []
    sentences = []
    
    for root, dirs, files in os.walk(validation_dir):
        for file in files:
            file_paths.append(os.path.join(root, file))
    
    for doc in file_paths:
        sentences.extend(sent_tokenize(open(doc).read()))
        
    return sentences

def get_sentence_from_user():
    sentences = []
    sentences.append(input("Provide a sentence to analyze >> "))
    return sentences

def get_fileID_from_user(fileID):
    sentences = []
    
    # Iterate through each category and get the file IDs to verify validitiy
    all_file_ids = []
    for category in reuters.categories():
        file_ids_in_category = reuters.fileids(category)
        all_file_ids.extend(file_ids_in_category)
    
    while True:
        if fileID in all_file_ids:
            sentences.extend(sent_tokenize(reuters.raw(fileID)))
            break
        else:
            fileID = input("Invalid file ID. Provide a valid file ID available in reuters corpus: >> ")

    return sentences
    
# Program start
print("COMP-6751 Project 3 Fall 2023. Iymen Abdella 40218280\n")
grammar, sentences = get_pipeline_options()

# Runs pipeline based on user provided options, reprompts for more text afterwards
while True:
    driver = PipelineDriver(sentences=sentences, grammar=grammar)
    driver.run()
    response = input("Would you like to process another text? (Y/N) ")
    while True:
        if response.lower().startswith('y'):
            grammar, sentences = get_pipeline_options()
            break
        elif response.lower().startswith('n'):
            print("exiting program...")
            exit()
        else:
            response = input("Invalid entry. Would you like to process another text? (Y/N) ")
