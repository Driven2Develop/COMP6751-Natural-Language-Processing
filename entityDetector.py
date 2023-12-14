import regex as re

import nltk
nltk.download('punkt') #tokenizer
nltk.download('averaged_perceptron_tagger') #POS tagger
nltk.download('maxent_ne_chunker')
nltk.download('words')

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk

class measuredEntityDetector:
    def __init__(self):
        self.patterns = [r'\d+', r'\d{1,3}(?:,\d{3})+', r'\d+\.\d+']
    
    # measured entity detection in sentence, rather than individual words.
    # returns a list of measured entities detected within the sentence detected by regex. 
    def detect_with_pattern(self, sentence):
        
        matches = []
        for pattern in self.patterns:
            matches.extend(re.findall(pattern, sentence))
        
        return self.annotate_measured_entities(sentence, matches)
        
    #uses POS for detection. 
    def detect_with_POS(self, sentence):
        matches = []

        for word, pos in pos_tag(word_tokenize(sentence)):
            if pos == "CD":
                matches.append(word) 
        
        return self.annotate_measured_entities(sentence, matches)
    
    # annotates the matches so that they align with sentence structure. Critical for ensuring CoNLL format
    def annotate_measured_entities(self, sentence, matches):
        measured_ent = []
        words = word_tokenize(sentence)
        pos_tags = pos_tag(words)

        # if next word is a noun combine them, otherwise ignore.
        for i, word in enumerate(pos_tags):
            if word[0] in matches and pos_tags[i+1][1] in ['NN', 'NNS', 'NNP', 'CD'] : 
                measured_ent.append("{} {}".format(word[0], pos_tags[i+1][0]))
            else:
                measured_ent.append(None)

        return measured_ent
        
class namedEntityDetector:
    def __init__(self):
        pass
    
    # accepts a tuple of words with their associated POS
    # for example with nltk.pos_tag(words)
    # has support for non binary NE detection
    def get_ne(self, pos_words, binary=True):
        ne = []
        ent_list = self.extract_entities(ne_chunk(pos_words, binary=binary), binary)
        
        if binary:
            for word in [pos[0] for pos in pos_words]:
                if word in ent_list:
                    ne.append("NE")
                else:
                    ne.append(None)
        else:
            for word in [pos[0] for pos in pos_words]:
                if word in ent_list.keys():
                    ne.append(ent_list[word])
                else:
                    ne.append(None)

        return ne
    
    #helper method that returns a list of all the detected named entities from named entity tree
    def extract_entities(self, tree, binary):
        named_entities = []
        if binary: 
            if hasattr(tree, 'label') and tree.label() == 'NE':
                named_entities.append(' '.join([child[0] for child in tree]))
                for subtree in tree:
                    if type(subtree) == nltk.Tree:
                        named_entities.extend(self.extract_entities(subtree))
        else:
            for t in tree:
                if isinstance(t, nltk.Tree):
                    entity_label = t._label
                    entity_type = t[0]
                    named_entities.append((entity_label, entity_type))
            
            named_entities = dict(zip([pos[0] for pos in [pos[1] for pos in named_entities]], [pos[0] for pos in named_entities] ))

        return named_entities