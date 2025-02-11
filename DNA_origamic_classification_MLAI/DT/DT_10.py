import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import time

start_time = time.time()  # Record the start time

# Function to load images from folder
def load_images_from_folder(folder):
    images = []
    labels = []
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(os.path.join(subdir, file)).convert('RGB')
                img = img.resize((224, 224))
                images.append(np.array(img))
                label = os.path.basename(subdir)
                labels.append(label)
    return np.array(images), np.array(labels)

# Function to train and evaluate the model
def train_and_evaluate(seed, X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)

    # Build decision tree model
    clf = DecisionTreeClassifier(random_state=seed)

    # Train the model
    clf.fit(X_train, y_train)

    # Predict the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Specify image directories
base_image_directory = '../simulation'
TEM_image_directory = '../TEM'
image_path = '../predictTEM'

# Load image data
images, labels = load_images_from_folder(base_image_directory)

# Preprocess data: flatten image data into 1D vectors
n_samples, height, width, channels = images.shape
data = images.reshape((n_samples, height * width * channels))

# Convert labels to numerical values
label_to_index = {label: idx for idx, label in enumerate(np.unique(labels))}
indexed_labels = np.array([label_to_index[label] for label in labels])

# Train and evaluate the model with different seeds
seeds = range(10)
accuracies = []

for seed in seeds:
    accuracy = train_and_evaluate(seed, data, indexed_labels)
    accuracies.append(accuracy)
    print(f'Seed: {seed}, Accuracy: {accuracy}')

# Calculate the average accuracy and standard deviation
average_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

print(f'Average Accuracy: {average_accuracy}')
print(f'Standard Deviation of Accuracy: {std_accuracy}')

# Calculate error bars (standard error of the mean)
sem_accuracy = std_accuracy / np.sqrt(len(seeds))
print(f'Standard Error of the Mean Accuracy: {sem_accuracy}')

# Save the trained model with the best seed
best_seed = seeds[np.argmax(accuracies)]
best_clf = DecisionTreeClassifier(random_state=best_seed)
X_train, X_test, y_train, y_test = train_test_split(data, indexed_labels, test_size=0.2, random_state=best_seed)
best_clf.fit(X_train, y_train)
joblib.dump(best_clf, 'decision_tree_model_best_seed.pkl')

# Load and use the best model for prediction
def prepare_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image).reshape(1, -1)  # Flatten to 1D vector and add batch dimension
    return image

def predict_image(image_path, model):
    image_vector = prepare_image(image_path)
    prediction = model.predict(image_vector)
    class_name = list(label_to_index.keys())[prediction[0]]
    return class_name

# Example usage
image_files = sorted([f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))])

# Load the best model
best_clf = joblib.load('decision_tree_model_best_seed.pkl')

# Process each image
for img_file in image_files:
    img_path = os.path.join(image_path, img_file)
    class_name = predict_image(img_path, best_clf)
    print(f'Image: {img_file}, Predicted class: {class_name}')

end_time = time.time()  # Record the end time
print(f"Running time: {end_time - start_time} seconds")