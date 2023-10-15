import regex as re

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
from nltk.parse import EarleyChartParser

# grammar definition
class contextFree:
    def __init__(self):
        self.patterns = [r'\d+', r'\d{1,3}(?:PUNC\d{3})+', r'\d+\.\d+']
        self.cfg = nltk.CFG.fromstring( """ 
            S -> CP RB PUNC PP PUNC NP PUNC WP VP IN CP PP PUNC RB VB DT JJ NP PP PP NP PUNC    
            S -> CP RB PUNC PP PUNC NP PUNC WP VP IN CP PP PUNC RB JJ CP PP PP NP PUNC    
            S -> CP RB PUNC PP PUNC NP PUNC WP VP IN CP PP PUNC RB VB DT JJ NP PP PP NP PUNC   
            S -> NP VP PP PUNC   
            S -> PP PUNC NP VP NP PUNC
            S -> NP PUNC WP VP PP PUNC VP NP PUNC   
            S -> PP PUNC NP VP IN  PRP NP PUNC  
            S -> JJ NP PUNC PP PUNC NP RB VP PP PP PUNC  
            S -> NP NP JJR IN CP NP NP IN CD CC IN CP NP PUNC  
            S -> JJ NP JJR IN CP CC JJ NP PP JJR IN CP  
            S -> NP NP JJR IN CD CC JJR IN CP CC RB CP NP NP CC NP PUNC
            S -> NP NP JJR IN CP PUNC 
            S -> NP NP IN CD CC IN CP NP PUNC
            S -> JJ NP JJR IN CP CC JJ NP PP JJR IN CP PUNC                           

            NP -> IN PRP NP | DT NP | NN | NNP | NNS

            PP -> IN NP | TO PRP NP | IN PRP NP
                                                                        
            VP -> VB | VBD | VBP | VP JJ | VP NP

            CP -> CD | CD NP | CD JJ NP | CD JJ
                                       
            NNP -> 'John' | 'Heart' | 'Body' | 'Celsius' | 'CO2' | 'C02' | 'Leukocyte' | 'Monday' | 'Tuesday' | 'Wednesday' | 'Thursday' | 'Friday' | 'Saturday' | 'Sunday'                                       
            CD -> '5' | '90' | '38' | '36' | '20' | '32' | '12000' | '4000' | '10' | 'one' | 'Two' | 'three' | 'four' | 'five' | 'six' | 'seven' | 'eight' | 'nine'
            PUNC -> ',' | '.'
            CC -> 'or'
            DT -> 'the' | 'a' | 'all' | 'an'   
            IN -> 'on' | 'for' | 'until' | 'from' | 'at' | 'in' | 'from' | 'than' | 'over' | 'under' | 'of' | 'On'
            JJ -> 'red' | 'sick' | 'Last' | 'Respiratory' | 'partial' | 'mmHg' | 'ate'                                                                                             
            JJR -> 'greater' | 'less'
            NN -> 'apple' | 'fridge' | 'table' | 'office' | 'week' | 'rate' | 'beats/minute' | 'temperature' | 'pressure' | 'count' | '%' | 'immature' | 'breaths/minute' | 'Body' | 'mmHg'
            NNS -> 'degrees' | '/microliters' | 'forms' | 'bands' | 'days' | 'apples'
            PRP -> 'his'   
            RB -> 'finally' | 'over' | 'ago' | 
            TO -> 'to'                           
            VB -> 'ate'
            VBD -> 'was' | 'took'                           
            VBP -> 'ate'
            WP -> 'who'                           
            """)
        
        self.parser = EarleyChartParser(self.cfg, trace=0)

    #prints out the tree format of parsed sentence using grammar
    def display(self, sentence):
        words = word_tokenize(sentence)

        #print("POS Tags: {}".format(pos_tag(words)))

        parses = list(self.parser.chart_parse(words).parses(self.cfg.start()))

        if len(parses) > 0:
            print("Displaying parsed tree:")
            for tree in parses:
                tree.pretty_print()
        else:
            print("Invalid Entry. Sentence could not be parsed.")