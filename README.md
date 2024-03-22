# WordsofWar
Analyze a corpus of speeches given by US presidents in the lead-up to major wars to predict if a president is preparing the nation for war.

TBD...

# Manifest:

## Python Module Files (helper functions, classes)
`BertSeqVect.py`

## Jupyter Notebooks

`Cleaning_Data.ipynb`

The Jupyter Notebook contains the code used to clean the input data (speeches.csv).

`EDA.ipynb`

This Jupyter Notebook contains code and visualizations for exploratory data analysis.

`Exploring_and_BasicNN.ipynb`

This Jupyter Notebook contains code that explores the data using some the pre-trained BERT model and vectorizer (see BertSeqVect.py). We also experiment with three models: (1) CNN, (2) gated RNN (LSTM), and (3) gated RNN (LSTM) with attention mechanism. After developing these models, we then begin exploring various ways to perform interpretable learning and try and discern how the last model differentiatese the two classes.

## Data Files

`Speeches_War_Clean.csv`

This Excel (comma-separated) file contains the cleaned data used for modeling.

`speeches.csv`

This Excel (comma-separated) file contains the original source data.

## License and ReadME

`LICENSE`

`README.md`
