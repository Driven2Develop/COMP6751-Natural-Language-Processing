# NLA Project
The purpose of the project is to create a natural language processing pipeline for interpreting and processing natural text such as: articles, academic reports, and other text exerts. 
The entire pipeline is built in python with support from the NLTK library. 

Each subsequent project p1 -> p2 -> p3 are all iterative improvements on the processing pipeline. New features are added and refinements are made to make the pipeline more accurate and capture the meaning of sentences more consistently. 

## P1
* First iteration of pipeline which includes features such as
  * measured entity detection and annotation
  * named entity detection and annotation (binary)
  * gazetteer interpretation

## P2
* Adds Context free grammar support for generative grammars parsed with Earley parsing. 
* refines named entity detection for support with multiple entity types
* Enforces CoNLL structure for sentences during outputs
