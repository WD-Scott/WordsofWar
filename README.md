[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<img alt="Static Badge" src="https://img.shields.io/badge/Repo_Status%3A-Work_in_Progress-blue?style=plastic&labelColor=%23232D4B&color=%23E57200">

# Words of War
In political discourse and geopolitical analysis, national leaders' words hold profound significance, often serving as harbingers of pivotal historical moments. From impassioned rallying cries to calls for caution, presidential speeches preceding major conflicts encapsulate the multifaceted dynamics of decision-making at the apex of governance. This project aims to use deep learning techniques to decode the subtle nuances and underlying patterns of US presidential rhetoric that may signal US involvement in major wars. While accurate classification is desirable, we seek to take a step further and identify discriminative features between the two classes (i.e., interpretable learning).

Through an interdisciplinary fusion of machine learning and historical inquiry, we aspire to unearth insights into the predictive capacity of neural networks in discerning the preparatory rhetoric of US presidents preceding war. Indeed, as the venerable Prussian General and military theorist Carl von Clausewitz admonishes, "War is not merely an act of policy but a true political instrument, a continuation of political intercourse carried on with other means."

# Manifest:
<details>
<summary><img src="images/py.png" align="left" width="40" height="40" /> Python Module Files (helper functions, classes)</summary>
  
- ### [`BertSeqVect.py`](https://github.com/WD-Scott/WordsofWar/blob/main/Python_Modules/BertSeqVect.py)

This Python module file includes the `BertSequenceVectorizer` class, which we designed to convert input text into vector representations using a pre-trained the Bidirectional Encoder Representations from Transformers (BERT) model.

  * Features:
    
    **BERT-based Vectorization**: Utilizes a pre-trained BERT model to generate vector representations of input text.
    
    **Tokenization**: Employs the BERT tokenizer to tokenize input text before vectorization.
    
    **Customizable Sequence Length**: Allows customization of the maximum length of input sequences for vectorization.
 
  * Usage
    
    Upon instantiation of the `BertSequenceVectorizer` object, the class automatically loads a pre-trained BERT model (bert-base-uncased by default) and its corresponding tokenizer, specifying the maximum length of input sequences for vectorization.

- ### [`plot_history.py`](https://github.com/WD-Scott/WordsofWar/blob/main/Python_Modules/plot_history.py)

This Python module file contains a helper function for plotting model history (accuracy, validation accuracy, loss, and validation loss).
    
</details>
<br>
<details>
<summary><img src="images/ipynb.png" align="left" width="40" height="40" /> Jupyter Notebooks</summary>

- ### [`Cleaning_Data.ipynb`](https://github.com/WD-Scott/WordsofWar/blob/main/Jupyter_Notebooks/Cleaning_Data.ipynb)

The Jupyter Notebook contains the code we used to clean the input data (speeches.csv) and set up the training, testing, and validation sets. In this notebook, we use the pre-trained BERT model and vectorizer (see BertSeqVect.py) to tokenize and vectorize the text data.

- ### [`EDA.ipynb`](https://github.com/WD-Scott/WordsofWar/blob/main/Jupyter_Notebooks/EDA.ipynb)

This Jupyter Notebook contains code and visualizations from our exploratory data analysis.

- ### [`Modeling.ipynb`](https://github.com/WD-Scott/WordsofWar/blob/main/Jupyter_Notebooks/Modeling.ipynb)

This Jupyter Notebook contains our code for the modeling experiments. We experiment with three models: (1) MLP, (2) gated RNN (LSTM), and (3) the same second model but with Attention mechanisms. After developing these models, we use the third approach to begin exploring various ways to perform interpretable learning to discern how the model differentiates the two classes. Eventually, we will incorporate a fourth approach; a pre-trained transformer.
</details>
<br>
<details>
<summary><img src="images/csv.png" align="left" width="40" height="40" /> Data Files</summary>

- ### [`Speeches_War_Clean.csv`](https://github.com/WD-Scott/WordsofWar/blob/main/Data_Files/Speeches_War_Clean.csv)

This file contains the cleaned data that we use for modeling.

- ### [`presidential_speeches.csv`](https://github.com/WD-Scott/WordsofWar/blob/main/Data_Files/presidential_speeches.csv)

This file contains the original source data.

- ### [`X_test.csv`](https://github.com/WD-Scott/WordsofWar/blob/main/Data_Files/X_test.csv)

This file contains the testing features (the vector representations of the input text).

- ### [`X_train.csv`](https://github.com/WD-Scott/WordsofWar/blob/main/Data_Files/X_train.csv)

This file contains the training features (the vector representations of the input text).

- ### [`X_val.csv`](https://github.com/WD-Scott/WordsofWar/blob/main/Data_Files/X_val.csv)

This file contains the validation features (the vector representations of the input text).

- ### [`y_test.csv`](https://github.com/WD-Scott/WordsofWar/blob/main/Data_Files/y_test.csv)

This file contains the testing labels (binary response variable 'War').

- ### [`y_train.csv`](https://github.com/WD-Scott/WordsofWar/blob/main/Data_Files/y_train.csv)

This file contains the training labels (binary response variable 'War').

- ### [`y_val.csv`](https://github.com/WD-Scott/WordsofWar/blob/main/Data_Files/y_val.csv)

This file contains the validation labels (binary response variable 'War').
</details>
