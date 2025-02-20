# Clonal Selection Algorithm for Spam Detection

This project implements a Clonal Selection Algorithm (CSA) to classify emails as spam or non-spam using the SpamAssassin dataset. The model is evaluated using standard classification metrics and visualizations, including the ROC curve, confusion matrix, and fitness evolution plot.

### The project consists of the following key components:

- best_implementation.py: Loads and preprocesses the dataset and implements CSA.
- evaluation.py: Evaluates the CSA model and generates visualizations.
- SpamAssassin/: Directory where the dataset must be placed.
- Outputs: Folder for storing generated visualizations such as ROC curves.

### Installation & Setup
Ensure Python is installed (recommended version: Python 3.8+), then install the required dependencies using:
```pip install -r requirements.txt```

If requirements.txt is missing, install the necessary libraries manually:
````
pip install numpy pandas scikit-learn matplotlib seaborn
````

### Download dataset
Download & Place the Dataset
The dataset is too large to be stored in this repository and must be manually downloaded from Kaggle.

Download the SpamAssassin Public Corpus from the following link:
SpamAssassin Dataset on Kaggle: https://www.kaggle.com/datasets/beatoa/spamassassin-public-corpus?resource=download-directory&select=easy_ham 

After downloading, extract and place the following directories inside the project’s SpamAssassin/ folder:

- easy_ham/ (Legitimate emails)
- hard_ham/ (Difficult-to-classify legitimate emails)
- spam_2/ (Spam emails)
- The dataset must be placed inside SpamAssassin/, ensuring the scripts can access and process it correctly.

## How to Run the Project
1. Train & Evaluate the Model
  - Run the CSA model using: `python best_implementation.py`

2. Generate Evaluation Metrics & Visualizations
  - After training, execute: `python evaluation.py`


