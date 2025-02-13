import torch
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
from PIL import Image
import os
import zipfile
from pathlib import Path
import random
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision.models.resnet import ResNet34_Weights
#from torchvision.models.resnet import ResNet34_Weights
#from torchvision.models.resnet import ResNet50_Weights
#from torchvision.models.resnet import ResNet101_Weights
#from torchvision.models.resnet import ResNet152_Weights

import time
start_time = time.time()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)
def estimate_model_size_in_MB(model):
    param_size = sum(p.numel() for p in model.parameters()) * 4  # 4 bytes for a 32-bit number
    return param_size / (1024 * 1024)  # Convert to megabytes

def load_random_image_from_category(base_directory):
    """
    Loads a random image from a random category within the specified base directory.
    Args:
    base_directory (str): The path to the base directory containing category subdirectories.

    Returns:
    PIL.Image: A PIL Image object of the randomly selected image.
    """
    # List all subdirectories (categories)
    categories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

    # Select a random category
    selected_category = random.choice(categories)

    # Build the path to the selected category
    category_path = os.path.join(base_directory, selected_category)

    # List all image files in the selected category
    image_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Select a random image file
    random_image_file = random.choice(image_files)

    # Load and return the image
    image_path = os.path.join(category_path, random_image_file)
    return Image.open(image_path)

# Specify your base image directory here
base_image_directory = '../../simulation'
TEM_image_directory = '../../TEM'
image_path='../../predictTEM'
bsize=32
num_epochs = 90
num_epochs_tune = 90

# Load a random image from one of the categories
random_image = load_random_image_from_category(base_image_directory)

# Display the image (optional)
random_image.show()

plt.imshow(random_image)
plt.axis('off')  # Turn off axis numbers and labels
plt.show()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),          # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean and std
                         std=[0.229, 0.224, 0.225])
])

#transform = transforms.Compose([
#    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
#    transforms.Resize(512),
#    transforms.CenterCrop(448),
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485], std=[0.229]),  # Adjust for a single channel
#])

# Load the dataset
dataset = datasets.ImageFolder(root=base_image_directory, transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsize, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsize, shuffle=True)

# Load pre-trained ResNet with different deepth
resnetNN = models.resnet34(weights=ResNet34_Weights.DEFAULT)
#resnetNN = models.resnet34(weights=ResNet34_Weights.DEFAULT)
#resnetNN = models.resnet50(weights=ResNet50_Weights.DEFAULT)
#resnetNN = models.resnet101(weights=ResNet101_Weights.DEFAULT)
#resnetNN = models.resnet152(weights=ResNet152_Weights.DEFAULT)

num_classes = len(dataset.classes)
resnetNN.fc = torch.nn.Linear(resnetNN.fc.in_features, num_classes)

criterion = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(alexnet.parameters(), lr=0.001, momentum=0.9)
#optimizer = torch.optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
#optimizer = torch.optim.SGD(googlenet.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.SGD(resnetNN.parameters(), lr=0.001, momentum=0.9)
# Define scheduler
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

torch.cuda.is_available()

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#alexnet.to(device)
#vgg16.to(device)
resnetNN.to(device)

import csv
# Open a CSV file to save the results
with open("training_results_res34_v2.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(["Epoch", "Training Loss", "Training Accuracy", "Validation Loss", "Validation Accuracy"])

    # Training loop
    for epoch in range(num_epochs):
        resnetNN.train()  # Set the model to training mode
        running_loss = 0.0  # Initialize running loss for the current epoch
        correct_train = 0  # Initialize the number of correct predictions for training
        total_train = 0  # Initialize the total number of samples for training

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device

            optimizer.zero_grad()  # Zero the parameter gradients

            outputs = resnetNN(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the weights

            running_loss += loss.item()  # Accumulate the loss

            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
            total_train += labels.size(0)  # Accumulate the total number of samples
            correct_train += (predicted == labels).sum().item()  # Accumulate the number of correct predictions

        # Calculate and print average loss and accuracy for the training epoch
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        print(f"Epoch {epoch+1}, Training Loss: {train_loss}, Training Accuracy: {train_accuracy}%")

        # Validate the model
        resnetNN.eval()  # Set the model to evaluation mode
        running_val_loss = 0.0  # Initialize running loss for validation
        correct_val = 0  # Initialize the number of correct predictions for validation
        total_val = 0  # Initialize the total number of samples for validation

        with torch.no_grad():  # Disable gradient computation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device
                outputs = resnetNN(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute the loss
                running_val_loss += loss.item()  # Accumulate the validation loss

                _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
                total_val += labels.size(0)  # Accumulate the total number of samples
                correct_val += (predicted == labels).sum().item()  # Accumulate the number of correct predictions

        # Calculate and print average loss and accuracy for the validation epoch
        val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%")

        # Save the results to the CSV file
        writer.writerow([epoch+1, train_loss, train_accuracy, val_loss, val_accuracy])

print("Training results have been written to training_results.csv")

#torch.save(alexnet.state_dict(), './alexnet_trained.pth')
torch.save(resnetNN.state_dict(), './resnet34_trained.pth')

# Assuming the necessary imports are done and the device is set

# Load pre-trained ResNet
resnetNN = models.resnet34(weights=ResNet34_Weights.DEFAULT)
#resnetNN = models.resnet34(weights=ResNet34_Weights.DEFAULT)
#resnetNN = models.resnet50(weights=ResNet50_Weights.DEFAULT)
#resnetNN = models.resnet101(weights=ResNet101_Weights.DEFAULT)
#resnetNN = models.resnet152(weights=ResNet152_Weights.DEFAULT)

resnetNN.fc = torch.nn.Linear(resnetNN.fc.in_features, num_classes)

state_dict = torch.load('./resnet34_trained.pth')

resnetNN.load_state_dict(state_dict)
resnetNN.eval()  # Start with eval mode

num_new_classes = 6  # Replace with your new number of classes

resnetNN.fc = torch.nn.Linear(resnetNN.fc.in_features, num_new_classes)

# Freeze all layers first
for param in resnetNN.parameters():
    param.requires_grad = False

# Unfreeze the last classifier layer
#for param in vgg16.classifier[6].parameters():
#    param.requires_grad = True
for param in resnetNN.fc.parameters():
    param.requires_grad = True

# Assuming you have the transforms and ImageFolder setup as previously shown
new_dataset = datasets.ImageFolder(root=TEM_image_directory, transform=transform)
new_train_loader = DataLoader(new_dataset, batch_size=bsize, shuffle=True)

#optimizer = torch.optim.Adam(vgg16.classifier[6].parameters(), lr=0.001)
optimizer = torch.optim.Adam(resnetNN.fc.parameters(), lr=0.001)

resnetNN.to(device)
resnetNN.train()

for epoch in range(num_epochs_tune):  # Fewer epochs needed for fine-tuning
    for inputs, labels in new_train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = resnetNN(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(new_train_loader)}")

# Save the fine-tuned model's state dictionary
torch.save(resnetNN.state_dict(), './fine_tuned_resnet34.pth')

# For loading the fine-tuned VGG16 model structure
resnetNN = models.resnet34(weights=ResNet34_Weights.DEFAULT)
#resnetNN = models.resnet34(weights=ResNet34_Weights.DEFAULT)
#resnetNN = models.resnet50(weights=ResNet50_Weights.DEFAULT)
#resnetNN = models.resnet101(weights=ResNet101_Weights.DEFAULT)
#resnetNN = models.resnet152(weights=ResNet152_Weights.DEFAULT)

# If you changed the input channel configuration, adjust the first conv layer accordingly (optional)

# Modify the final layer to match the number of classes
resnetNN.fc = nn.Linear(resnetNN.fc.in_features, num_new_classes)

# Load your custom trained weights
state_dict = torch.load('./fine_tuned_resnet34.pth')

resnetNN.load_state_dict(state_dict)
resnetNN.eval()  # Set the model to evaluation mode

# Model parameters and size
total_params, trainable_params = count_parameters(resnetNN)
model_size = estimate_model_size_in_MB(resnetNN)
print(f"Total Parameters: {total_params}")
print(f"Trainable Parameters: {trainable_params}")
print(f"Estimated Model Size (MB): {model_size:.2f}")

# Load and transform an image

def prepare_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension


def predict_image(image_path):
    image_tensor = prepare_image(image_path)

    # Predict
    with torch.no_grad():  # No need to compute gradients
        outputs = resnetNN(image_tensor)
        _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability

        # Optionally, convert the index to a class name, depending on your dataset
        # class_name = index_to_class_name(predicted.item())

    return predicted.item()
# Replace this with your actual mapping
class_names = {
    0: '1QD-1origami',
    1: '1QD-2origami',
    2: '1QD-3origami',
    3: '1QD-4origami',
    4: '1QD-5origami',
    5: '1QD-6origami'
}

# List and sort the image files
image_files = sorted([f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))])

# Process each image
for img_file in image_files:
    img_path = os.path.join(image_path, img_file)
    prediction = predict_image(img_path)
    class_name = class_names[prediction]
    print(f'Image: {img_file}, Predicted class: {class_name}')


# In[27]:

def predict_image_with_probabilities(image_path):
    image_tensor = prepare_image(image_path)

    with torch.no_grad():  # No need for gradients
        outputs = resnetNN(image_tensor)

        # Apply softmax to convert logits to probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        probability_values, predicted_classes = torch.max(probabilities, 1)  # Get the highest probability and its corresponding class

        return probabilities.squeeze().numpy(), predicted_classes.item(), probability_values.item()

# Example usage
image_path = os.path.join(image_path,'unknown02.png')
probabilities, predicted_class, probability_value = predict_image_with_probabilities(image_path)
print(f'Predicted class index: {predicted_class}, with probability: {probability_value}')
print(f'Probabilities for all classes: {probabilities}')


# In[ ]:

end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime of the script: {total_time} seconds")




