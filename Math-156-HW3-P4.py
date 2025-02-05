import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = load_breast_cancer()
X, y = data.data, data.target # Define features and lables

# Split data into 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Split temporary in to 15% validation, 15% test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Normalize features using StandardScaler (zero mean, unit variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit to training data and transform
X_val = scaler.transform(X_val)          # Transform validation data
X_test = scaler.transform(X_test)        # Transform test data


# Report dataset sizes
print("Training set size:", X_train.shape)
print("Validation set size:", X_val.shape)
print("Test set size:", X_test.shape)

# Count class occurrences
unique_train, counts_train = np.unique(np.concatenate((y_train, y_val)), return_counts=True)

# Report class occurrences
print("Class Distribution in Training & Validation Sets:")
for clss, count in zip(unique_train, counts_train):
    print("Class", clss, ":", count, "samples")
    
# Train Binary Log. Reg. with Mini-Batch SGD from P3

# Sigmoid function
def sigmoid(z):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-z))

# Binary cross-entropy loss function
def CE_loss(y, t):
    """Compute binary cross-entropy loss."""
    return -np.mean(t * np.log(y + 1e-9) + (1 - t) * np.log(1 - y + 1e-9)) # Add 1e-9 to avoid numerical errors

# Function to make predictions
def predict(X, weights):
    """Compute predictions using the logistic regression model."""
    return (sigmoid(X @ weights) >= 0.5).astype(int)

# Mini-batch SGD for logistic regression
def mini_batch_sgd(X, t, batch_size=64, learning_rate=0.01, max_iter=1000):
    """
    Train a logistic regression model using mini-batch SGD.

    Parameters:
    X : ndarray (N, D) - Input features
    t : ndarray (N, 1) - Target labels (0 or 1)
    batch_size : int - Number of samples per batch
    learning_rate : float - Learning rate for gradient descent
    max_iters : int - Number of training iterations

    Returns:
    w : ndarray (D, 1) - Optimized weight vector
    """
    N, D = X.shape # Number of samples and features
    w = np.random.randn(D) # Initialize weights from standard Gaussian distribution.
    
    for iteration in range(max_iter):
        indices = np.random.permutation(N) # Shuffle data
        X_shuffled, t_shuffled = X[indices], t[indices]
        
        for i in range(0, N, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            t_batch = t_shuffled[i:i + batch_size]
            
            y_batch = sigmoid(X_batch @ w) # Compute predictions
            gradient = (X_batch.T @ (y_batch - t_batch)) / batch_size # Compute gradient
            w -= learning_rate * gradient # Update weights
            
    return w
            
# Train model
weights = mini_batch_sgd(X_train, y_train)

# Make predictions on the test set
y_test_pred = predict(X_test, weights)

# Evaluate the model
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print("Accuracy:", accuracy)
print("precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)










