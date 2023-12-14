from contextlib import redirect_stdout
import pandas as pd
import os
import time
from datetime import datetime

import nltk
nltk.download('reuters') #corpus
nltk.download('punkt') #tokenizer
nltk.download('averaged_perceptron_tagger') #POS tagger
nltk.download('maxent_ne_chunker')
nltk.download('words')

from nltk.tokenize import word_tokenize
from nltk import pos_tag

from gazzeteer import Gazetteer
from entityDetector import namedEntityDetector
from entityDetector import measuredEntityDetector
from sentiment import Sentiment_Analyzer
from collections import Counter

# Class definition, has an instance of each of the other modules
class PipelineDriver:
    def __init__(self, sentences=None, grammar=None):
        self.grammar = grammar
        self.sentences = sentences
        self.coNLL_headers = ['Word','POS_tag','Named_Entity','Entity','Measured_Entity']
        self.output_timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        self.gazetteer = Gazetteer()
        self.named_entity = namedEntityDetector()
        self.measured_entity = measuredEntityDetector()
        self.sentiment = Sentiment_Analyzer() # P3
 
    def run(self):
        output_response = input("Would you like to save the output to a file? (y/n) ")
        sentiment_stance = []
        for sent in self.sentences:
            
            print(f"Processing sentence >> {sent} <<")
            tree_bracket, tree_pretty, overall_sentiment, output = self.process_sentence(sent)
            
            if output_response.lower().startswith('y'):
                sentiment_stance.append(self.save_output(sent, tree_bracket, tree_pretty, overall_sentiment, output))
                print(f"Finished processing sentence >> {sent} << Saved Locally.")
            else:
                print(f"Finished processing sentence >> {sent} <<")

        # logic to calculate the overall stance of a paragraph. If both positive and negative are 0, then neutral, otherwise takes the max of positive and negative
        if output_response.lower().startswith('y'):
            max_sentiment = 'neutral' # default behavior, also covers the case when positive and negative sentiments are equal
            if (sentiment_stance.count('positive') == 0 and sentiment_stance.count('negative')==0):
                max_sentiment = 'neutral'
            else:
                if (sentiment_stance.count('positive') > sentiment_stance.count('negative')):
                    max_sentiment = 'positive'
                else:
                    max_sentiment = 'negative'

            self.write_tofile(os.path.join(os.path.join(os.path.dirname(__file__), "output", self.output_timestamp), "Overall_sentiment.txt"),
                "Overall stance of paragraph is {}.\nThe total sentiment count from all the processed sentences is:\nPositive: {}\tNegative: {}\tNeutral: {}".format(
                    max_sentiment, sentiment_stance.count('positive'), sentiment_stance.count('negative'), sentiment_stance.count('neutral')
                )
            )
        
        print("All sentences processed.")

    def save_output(self, sent, tree_bracket, tree_pretty, overall_sentiment, output):
        file_dir = os.path.join(os.path.dirname(__file__), "output", self.output_timestamp)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        self.write_tofile(os.path.join(file_dir, "Tree_bracket.txt"), tree_bracket)
        self.write_tofile(os.path.join(file_dir, "Tree_pretty.txt"), tree_pretty)

        # saves the overall sentiment to a file. If the sentence cannot be parsed, then the sentiment defaults to neutral.
        keys_with_max_value = [key for key, value in overall_sentiment.items() if value == max(overall_sentiment.values())]
        senti_output = "Overall Sentiment of sentence >> {} << is {}.\nThe total sentiment count from all the produced trees is:\nPositive: {}\tNegative: {}\tNeutral: {}".format(
            sent, keys_with_max_value[0] if len(keys_with_max_value) == 1 else 'neutral', overall_sentiment['positive'], overall_sentiment['negative'], overall_sentiment['neutral']
            )
        
        self.write_tofile(os.path.join(file_dir, "Overall_sentiment.txt"), senti_output)
        
        # coNLL output
        df = pd.DataFrame({self.coNLL_headers[i]: output[i] for i in range(len(output))})
        if not os.path.exists(os.path.join(file_dir, "coNLL.txt")):
            df.to_csv(os.path.join(file_dir, "coNLL.txt"), index=True, sep='\t')
        else:
            df.to_csv(os.path.join(file_dir, "coNLL.txt"), index=True, sep='\t', mode='a', header=False)

        return keys_with_max_value[0] if len(keys_with_max_value) == 1 else 'neutral' # return the key of the most frequent
    
    # helper method for writing information to a file as output
    def write_tofile(self, file_path, content):
        separator = "\n\n<<" + "-" * 100 + ">>\n\n"

        with open(file_path, 'a') as file:
            if isinstance(content, str):
                file.write(content)
            else:
                for tree in content:
                    file.write(str(tree) + "\n")
            
            file.write(separator)

    # processes sentences according to user choice 
    def process_sentence(self, sentence):
        words = word_tokenize(sentence)
        pos_words = pos_tag(words)
        named_ent = self.named_entity.get_ne(pos_words, False)
        gaz_ent = [self.gazetteer.annotate_gazetteer(word) for word in words]
        measured_ent = self.measured_entity.detect_with_pattern(sentence)

        # process using grammar
        print(f"POS words >> {pos_words}")
        tree_bracket, tree_pretty, overall_sentiment = self.grammar.display(sentence)

        # matches the headers within self.coNLL_headers
        return tree_bracket, tree_pretty, overall_sentiment, [words, [pos[1] for pos in pos_tag(words)], named_ent, gaz_ent, measured_ent]