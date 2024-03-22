# Words of War
Analyze a corpus of speeches given by US presidents in the lead-up to major wars to predict if a president is preparing the nation for war.

TBD...

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Manifest:
<details>
<summary><img src="images/py.png" align="left" width="40" height="40" /> Python Module Files (helper functions, classes)</summary>
  
- ### `BertSeqVect.py`
</details>
<br>
<details>
<summary><img src="images/ipynb.png" align="left" width="40" height="40" /> Jupyter Notebooks</summary>

- ### `Cleaning_Data.ipynb`

The Jupyter Notebook contains the code used to clean the input data (speeches.csv).

- ### `EDA.ipynb`

This Jupyter Notebook contains code and visualizations for exploratory data analysis.

- ### `Exploring_and_BasicNN.ipynb`

This Jupyter Notebook contains code that explores the data using some the pre-trained BERT model and vectorizer (see BertSeqVect.py). We also experiment with three models: (1) CNN, (2) gated RNN (LSTM), and (3) gated RNN (LSTM) with attention mechanism. After developing these models, we then begin exploring various ways to perform interpretable learning and try and discern how the last model differentiatese the two classes.
</details>
<br>
<details>
<summary><img src="images/csv.png" align="left" width="40" height="40" /> Data Files</summary>

- ### `Speeches_War_Clean.csv`

This file contains the cleaned data used for modeling.

- ### `speeches.csv`

This file contains the original source data.
</details>
