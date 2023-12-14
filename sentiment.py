import nltk
from nltk.corpus import opinion_lexicon
from nltk.corpus import sentence_polarity
nltk.download('opinion_lexicon')
nltk.download('sentence_polarity')
'''
Consider adding a negation factor: w*-0.5 because sometimes the effect is dampened 
(however we can toggle this factor)
and a modality factor he MAY be hurt: w*0.5 the dampening factor can be changed based on word
could, would, should, may, likely, etc..
'''
class Sentiment_Analyzer:

    def __init__(self):
        self.negative_lexica = opinion_lexicon.negative()
        self.negative_lexica_size = len(self.negative_lexica)
        self.positive_lexica = opinion_lexicon.positive()
        self.positive_lexica_size = len(self.positive_lexica)
        self.sentiment_categories = sentence_polarity.categories()
