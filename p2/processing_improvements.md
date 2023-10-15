# improvements made to p2 preprocess compared with p1

## Gazetteers:
* **Unit Gazetteer:** has been sourced from an official unit gazetteer where the units have been annotated with POS tags
     * This should help identify gazetteer members better, and allows for additional analysis for additional POS tagging. 
     * The gazetteer also has significantly more entries from ~700 --> ~1500

## Entity Detection
* **Measured Entity:** previously we used regex for detecting measured entities, however we have supplemented our solution with POS 
     * Another option to detect measured entities using their POS instead of regex.

## Design
* **Pipe and filter Architecture:** To improve the flexibility and adaptability of the processing program, it has been refactored and re-implemented to prioritize a pipe and filter archicture. 
* **Decoupling of Features:** In an attempt to simplify the design and improve modularity, resuability, and future maintainability. Core features such as entity recognition have been moved to their own classes where all the behavior can be kept separate from the flow of code execution. 
* **Improved User Interaction:** the functionality of the program has expanded the usability to run validation texts stored locally, or directly from the reuters corpus continously and consistently. Additionally, all output is stored locally, but users have the option to display a sample of the data on the command line.  