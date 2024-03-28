[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Words of War
In political discourse and geopolitical analysis, national leaders' words hold profound significance, often serving as harbingers of pivotal historical moments. From impassioned rallying cries to calls for caution, presidential speeches preceding major conflicts encapsulate the multifaceted dynamics of decision-making at the apex of governance.

This project aims to use deep learning techniques to decode the subtle nuances and underlying patterns of US presidential rhetoric that may signal US involvement in major wars. Through an interdisciplinary fusion of machine learning and historical inquiry, we aspire to unearth insights into the predictive capacity of neural networks in discerning the preparatory rhetoric of US presidents preceding war. Indeed, as the venerable Prussian General and military theorist Carl von Clausewitz admonishes, "War is not merely an act of policy but a true political instrument, a continuation of political intercourse carried on with other means."

# Manifest:
<details>
<summary><img src="images/py.png" align="left" width="40" height="40" /> Python Module Files (helper functions, classes)</summary>
  
- ### `BertSeqVect.py`

This Python module file includes the `BertSequenceVectorizer` class, which is designed to convert input text into vector representations using a pre-trained the Bidirectional Encoder Representations from Transformers (BERT) model.

  * Features:
    
    **BERT-based Vectorization**: Utilizes a pre-trained BERT model to generate vector representations of input text.
    
    **Tokenization**: Employs the BERT tokenizer to tokenize input text before vectorization.
    
    **Customizable Sequence Length**: Allows customization of the maximum length of input sequences for vectorization.
 
  * Usage
    
    Upon instantiation of the `BertSequenceVectorizer` object, the class automatically loads a pre-trained BERT model (bert-base-uncased by default) and its corresponding tokenizer.     Additionally, it specifies the maximum length of input sequences for vectorization.
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
