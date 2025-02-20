import numpy as np
import pandas as pd
import re
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
import matplotlib.pyplot as plt


# Load SpamAssassin dataset from local directories
def load_spamassassin_data(base_path):
    data = []
    labels = []

    for category, label in [("easy_ham", 0), ("spam_2", 1)]:
        folder_path = os.path.join(base_path, category)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='latin-1') as file:
                email_content = file.read()
                email_content = re.sub(r"(Subject:|From:|To:|Date:).*?\n", "", email_content)  # Keep useful metadata
                data.append(email_content)
                labels.append(label)

    return pd.DataFrame({"text": data, "label": labels})


# Set path to extracted SpamAssassin dataset
spamassassin_path = os.path.join(os.getcwd(), "SpamAssassin")

# Load dataset
df = load_spamassassin_data(spamassassin_path)


# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
    return text


# Apply preprocessing
X_text = df["text"].astype(str).apply(preprocess_text).tolist()
y = df["label"].astype(int).values  # 1 = spam, 0 = non-spam

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
X = vectorizer.fit_transform(X_text)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Clonal Selection Algorithm (CSA) for Spam Detection
def clonal_selection_algorithm(X_train, y_train, X_test, y_test, population_size=100, mutation_rate=0.015,
                               generations=30):
    print("Training the Clonal Selection Algorithm... Please wait.")

    # Randomly initialise population of "antibodies" (classifier rules)
    antibodies = np.random.rand(population_size, X_train.shape[1])
    class_weights = {0: 1.0, 1: 1.1}  # Adjusted weight to balance spam detection
    labels = np.random.choice([0, 1], size=population_size, p=[class_weights[0] / sum(class_weights.values()),
                                                               class_weights[1] / sum(
                                                                   class_weights.values())])  # Initial random labels

    for generation in range(generations):
        scores = X_train @ antibodies.T  # Compute activation scores
        probabilities = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)  # Softmax scoring
        predictions = np.argmax(probabilities, axis=1)
        accuracy = (predictions == y_train).mean()

        # Select top antibodies based on performance
        top_indices = np.argsort(-accuracy)[:population_size // 2]
        antibodies = antibodies[top_indices]
        labels = labels[top_indices]

        # Clone and mutate
        elite_size = max(2, population_size // 7)  # Preserve top 15%
        elite_antibodies = antibodies[:elite_size]  # Elitism
        mutations = np.random.normal(scale=mutation_rate, size=antibodies.shape)
        antibodies = np.vstack([antibodies, antibodies + mutations])
        labels = np.hstack([labels, labels])

    # Evaluate final model
    scores = X_test @ antibodies.T
    probabilities = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='binary')
    print(
        f"Final Model - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, predictions)
    plt.plot(fpr, tpr, label=f"CSA (AUC={auc(fpr, tpr):.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.title("ROC Curve")
    plt.show()


# Run CSA for Spam Detection
print("Starting CSA execution...")
clonal_selection_algorithm(X_train, y_train, X_test, y_test)
