import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from best_implementation import X_train, y_train, X_test, y_test


def evaluate_model(y_test, predictions):
    """Evaluate model performance using standard classification metrics and visualisations."""
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='binary')
    cm = confusion_matrix(y_test, predictions)
    false_positive_rate = cm[0, 1] / (cm[0, 1] + cm[0, 0])  # FPR calculation

    print(
        f"Final Model - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}, FPR: {false_positive_rate:.2f}")

    # Confusion Matrix Plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Spam', 'Spam'],
                yticklabels=['Non-Spam', 'Spam'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, predictions)
    plt.plot(fpr, tpr, label=f"CSA (AUC={auc(fpr, tpr):.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.title("ROC Curve")
    plt.show()


def clonal_selection_algorithm(X_train, y_train, X_test, y_test, population_size=100, mutation_rate=0.015,
                               generations=30):
    print("Training the Clonal Selection Algorithm... Please wait.")

    antibodies = np.random.rand(population_size, X_train.shape[1])
    class_weights = {0: 1.0, 1: 1.1}  # Adjusted weight to balance spam detection
    labels = np.random.choice([0, 1], size=population_size, p=[class_weights[0] / sum(class_weights.values()),
                                                               class_weights[1] / sum(class_weights.values())])
    fitness_over_time = []  # Track model fitness over generations

    for generation in range(generations):
        scores = X_train @ antibodies.T  # Compute activation scores
        probabilities = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)  # Softmax scoring
        predictions = np.argmax(probabilities, axis=1)
        accuracy = (predictions == y_train).mean()
        fitness_over_time.append(accuracy)

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

    evaluate_model(y_test, predictions)

    # Evolution of Fitness Over Iterations
    plt.plot(range(1, generations + 1), fitness_over_time, marker='o')
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Accuracy)")
    plt.title("Evolution of CSA Model Over Generations")
    plt.show()


# Run CSA for Spam Detection
print("Starting CSA execution...")
clonal_selection_algorithm(X_train, y_train, X_test, y_test)
