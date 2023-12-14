import math
import regex as re

import nltk
nltk.download('words')

from nltk.tokenize import word_tokenize
from nltk import parse
from nltk.grammar import FeatureGrammar
from nltk.parse import EarleyChartParser, FeatureEarleyChartParser
from nltk.tree.prettyprinter import TreePrettyPrinter
from nltk.tree import Tree
from itertools import tee

# Baseline Afinn grammar
# Source: https://finnaarupnielsen.wordpress.com/2011/06/20/simplest-sentiment-analysis-in-python-with-af/
class AfinnBaseline:
    def __init__(self, filenameAFINN=None):
        self.afinn =  dict(map(lambda line: (line.split('\t')[0], int(line.split('\t')[1])), open(filenameAFINN)))

    # returns sentiment and valence, optional weight function, default is sqrt (N)
    def display(self, sentence, weight = math.sqrt):
        words = word_tokenize(sentence)
        sentiments = [self.afinn.get(word,0) for word in words]

        overall_sentiment = {
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }
        
        # Weight function is applied to sentiment sum as normalizing function, by default its sqrt(N), where N is number of words
        if sum(sentiments) != 0:
            sentiment = float(sum(sentiments))/weight(len(sentiments))
        else:
            sentiment = 0 # case when no sentiment is detected, default to neutral
        
        if sentiment == 0:
            to_return = f"Sentence >>{sentence}<< has NEUTRAL sentiment. Valence = 0."
        elif sentiment > 0:
            to_return = f"Sentence >>{sentence}<< has POSITIVE sentiment. Valence = {sentiment}."
            overall_sentiment['positive'] = sentiment
        else:
            # we use the absolute value of the negative sentiment so its inline with feature grammars sentiment measures
            to_return = f"Sentence >>{sentence}<< has NEGATIVE sentiment. Valence = {sentiment}"
            overall_sentiment['negative'] = abs(sentiment) 

        return to_return, to_return, overall_sentiment

# feature grammar definition
class FeatureGrammar:
    def __init__(self, grammar_path=None):
        self.grammar= nltk.grammar.FeatureGrammar.fromstring(open(grammar_path).read())
        self.parser = parse.FeatureEarleyChartParser(self.grammar)

    # remove the period as it interferes with parsing
    def display(self, sentence):

        if sentence.endswith('.'):
            sentence = sentence[:-1]

        words = word_tokenize(sentence)
        trees, trees_copy= tee(self.parser.parse(words))
        should_print = False
        error_message = "Ungrammatical entry: Unable to parse sentence using Feature Grammar."

        tree_bracket = []
        tree_pretty_print = []

        # returns the overall sentiment of a sentence. Defaults to neutral if the sentence is ungrammatical
        sentiments = []
        overall_sentiment = {
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }

        try:
            first_item = next(trees_copy)
            should_print = True;
        except StopIteration:
            print(error_message)
            return error_message + "\n>>" + sentence, error_message + "\n" + sentence, overall_sentiment 

        if should_print:
            for tree in trees:
                sentiments.append(tree.label()['SENTI'])
                tree_bracket.append(tree)
                pretty_tree = TreePrettyPrinter(Tree.fromstring(str(tree))).text()
                print(pretty_tree)
                tree_pretty_print.append(pretty_tree)

        overall_sentiment = {
            'positive': sentiments.count('positive'),
            'negative': sentiments.count('negative'),
            'neutral': sentiments.count('neutral')
        }
          
        return tree_bracket, tree_pretty_print, overall_sentiment

# cfg grammar definition
class ContextFree:
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
            S -> NP NP JJR IN CP NP NP IN CP CC IN CP NP PUNC  
            S -> JJ NP JJR IN CP CC JJ NP PP JJR IN CP  
            S -> NP NP JJR IN CP CC JJR IN CP CC RB CP NP NP CC NP PUNC
            S -> NP NP JJR IN CP PUNC 
            S -> NP NP IN CP CC IN CP NP PUNC
            S -> JJ NP JJR IN CP CC JJ NP PP JJR IN CP PUNC  
                                       
            S -> NP NP JJ NP NP VP JJR IN CP PP IN PRP VP NP PUNC  
            S -> NP VP CP PP TO NP IN CP JJ NP CC NP VP PRP PUNC  
            S -> RB PUNC NP PP VP RB CP PP CC PP PP PUNC  
            S -> NP PP VBN IN RB CP IN CP PP TO NP PUNC  
            S -> PRP NP VP VBN PP WDT VBZ DT JJ NP NP CC DT JJ NP NP WDT VBZ NP RB PUNC    

            NP -> IN PRP NP | DT NP | NN | NNP | NNS

            PP -> IN NP | TO PRP NP | IN PRP NP
                                                                        
            VP -> VB | VBD | VBP | VP JJ | VP NP

            CP -> CD | CD NP | CD JJ NP | CD JJ
                                       
            NNP -> 'John' | 'Heart' | 'Body' | 'Celsius' | 'CO2' | 'C02' | 'Leukocyte' | 'Monday' | 'Tuesday' | 'Wednesday' | 'Thursday' | 'Friday' | 'Saturday' | 'Sunday' | '’'                                      
            CD -> '5' | '90' | '38' | '36' | '20' | '32' | '12000' | '4000' | '10' | 'one' | 'Two' | 'two' | 'three' | 'four' | 'five' | 'six' | 'seven' | 'eight' | 'nine' | '6' | '8◦C' | '1.9◦C'
            PUNC -> ',' | '.' 
            CC -> 'or' | 'and'
            DT -> 'the' | 'a' | 'all' | 'an' | 'The'  
            IN -> 'on' | 'for' | 'until' | 'from' | 'at' | 'in' | 'from' | 'than' | 'over' | 'under' | 'of' | 'On' | 'after' | 'by'
            JJ -> 'red' | 'sick' | 'Last' | 'Respiratory' | 'partial' | 'mmHg' | 'ate' | 's' | 'last' | 'large' | 'small'                                                                                         
            JJR -> 'greater' | 'less'
            NN -> 'apple' | 'fridge' | 'table' | 'office' | 'week' | 'rate' | 'beats/minute' | 'temperature' | 'pressure' | 'count' | '%' | 'immature' | 'breaths/minute' | 'Body' | 'mmHg' | 'heart' | 'refrigerator' | 'fruit' | 'drawer' | 'dairy' | 'food' | '-5◦C'
            NNS -> 'degrees' | '/microliters' | 'forms' | 'bands' | 'days' | 'apples'| 'meters' | 'seconds'
            PRP -> 'his' | 'he' | 'it' | 'Our'
            RB -> 'finally' | 'over' | 'ago' | 'Sadly' | 'approximately' | 'perfectly'
            TO -> 'to'                           
            VB -> 'ate' | 'are'
            VBD -> 'was' | 'took' | 'rolled'    
            VBN -> 'decreased' | 'stored'                                            
            VBP -> 'ate'
            VBZ -> 'contains' | 'keep'                           
            WP -> 'who'
            WDT -> 'that'                                                      
            """)
        
        self.parser = EarleyChartParser(self.cfg, trace=0)

    #prints out the tree format of parsed sentence using grammar
    def display(self, sentence):
        words = word_tokenize(sentence)
        parses = list(self.parser.chart_parse(words).parses(self.cfg.start()))

        tree_bracket = []
        tree_pretty_print = []
        
        if len(parses) > 0:
            for tree in parses:
                tree_bracket.append(tree)
                tree_pretty_print.append(tree.pretty_print())
                tree.pretty_print()
        else:
            print("Ungrammatical entry: Unable to parse sentence using Context Free Grammar.")
        
        return tree_bracket, tree_pretty_print