import os
import pandas as pd
import regex as re

gaz_directory = "gaz/"

gaz_unit_file_name = "unit_gazetteer.csv"
gaz_country_file_name = "country_gazetteer.tsv"
gaz_currency_file_name = "country_gazetteer.tsv"

gaz_unit_file_path = os.path.join(os.path.dirname(__file__), os.path.join(gaz_directory, gaz_unit_file_name))
gaz_country_file_path =  os.path.join(os.path.dirname(__file__), os.path.join(gaz_directory, gaz_country_file_name))
gaz_currency_file_path = os.path.join(os.path.dirname(__file__), os.path.join(gaz_directory, gaz_currency_file_name))

class Gazetteer:
    def __init__(self):
        self.gaz_unit = dict(zip(pd.read_csv(gaz_unit_file_path, sep=',', header=0, usecols=['Unit'])['Unit'].tolist(), pd.read_csv(gaz_unit_file_path, sep=',', header=0, usecols=['POS'])['POS'].tolist())) 
        self.gaz_country =  pd.read_csv(gaz_country_file_path, sep='\t', header=0, usecols=['Country'])['Country'].tolist()
        self.gaz_currency = pd.read_csv(gaz_currency_file_path, sep='\t', header=0, usecols=['CurrencyName'])['CurrencyName'].tolist()
        
        self.gaz_country = [s.lower() for s in self.gaz_country]
        self.gaz_currency = [s.lower() for s in self.gaz_currency]
        # unit gazzeteer already lower cased

    # Checks if a word belongs to any gazetteer. If so returns the asssociated annotation.
    # ensures the word is singular, if plural, and in lower case.
    def annotate_gazetteer(self, word):
        word = word.lower()
        if word.endswith('s'):
            word = word[:-1]

        if '/' in word or '\\' in word:
            for token in re.split(r'[\\\/]', word):
                if token.endswith('s'):
                    token = token[:-1]
                if token in self.gaz_unit.keys():
                    return "Unit"

        if word in self.gaz_country:
            return "Country"
        elif word in self.gaz_currency:
            return "Currency"
        elif word in self.gaz_unit.keys() and self.gaz_unit[word] == 'CD' : #additional check for unit to make sure POS are aligned
            return "Unit"