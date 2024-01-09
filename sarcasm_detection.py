import sys
import torch
import random
import csv

from statistics import stdev
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sentence_transformers import SentenceTransformer

# set seeds for reproducibility
seed = 67
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# get source file from command line
source_path = sys.argv[1]

# create dictionaries with id and sequence / id and label
sequence_dict = {}
label_dict = {}

# open csv file and read lines
with open(source_path, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)

    # skip header
    next(csv_reader)
    
    # iterate over lines
    for line in csv_reader:
        # get id and sequence
        id = line[0]
        sequence = line[1]
        # get label
        label = line[2]
        # add id and sequence to dictionary
        sequence_dict[id] = sequence
        # add id and label to dictionary
        label_dict[id] = label

# set up embedding dictionary
embedding_dict = {}

# create embeddings for sequences
for id in sequence_dict:
    # get sequence
    sequence = sequence_dict[id]
    # compute embedding
    embedding = model.encode(sequence, convert_to_tensor=True)
    # add embedding to embedding dictionary
    embedding_dict[id] = embedding

# convert dictionaries to lists
ids = list(label_dict.keys())
labels = [label_dict[id] for id in ids]
embeddings = [embedding_dict[id] for id in ids]

# combine lists into tuples
data = list(zip(ids, embeddings, labels))

# shuffle data
random.shuffle(data)

# set up cross-validation
kf = KFold(n_splits=5, shuffle=True)

# set up lists to store accuracies, f1 scores, recalls and precisions
accuracies = []
f1_scores = []
recall_values = []
precision_values = []

# define number of epochs
num_epochs = 3

# cross-validation
for train_index, test_index in kf.split(data):
    # split data into training and test
    train_data = [data[i] for i in train_index]
    test_data = [data[i] for i in test_index]

    # set up linear layer
    linear_layer = torch.nn.Linear(384, 2)

    # set up optimizer
    optimizer = torch.optim.SGD(linear_layer.parameters(), lr=0.01)

    # set up loss
    loss = torch.nn.CrossEntropyLoss()

    # training
    for epoch in range(num_epochs):
        for id, embedding, label in train_data:
            # get label
            label_tensor = torch.tensor(1 if label == 'Yes' else 0)
            # get prediction
            prediction = linear_layer(embedding)
            # set gradients to 0
            optimizer.zero_grad()
            # compute loss
            loss = loss(prediction, label_tensor)
            # back propagation
            loss.backward()
            # update weights
            optimizer.step()

    # set up evaluation variables
    correct_predictions = 0
    total_predictions = 0
    predictions = []
    true_labels = []

    # set model to evaluation mode
    linear_layer.eval()

    # evaluation
    for id, embedding, label in test_data:
        # get label
        label_tensor = torch.tensor(1 if label == 'Yes' else 0)
        # get prediction
        with torch.no_grad():
            prediction = linear_layer(embedding)
        # get predicted class
        predicted_class = torch.argmax(prediction).item()
        # update evaluation variables
        correct_predictions += (predicted_class == label_tensor.item())
        total_predictions += 1
        # store predictions and true labels
        predictions.append(predicted_class)
        true_labels.append(label_tensor.item())

    # calculate f1 score, recall, and precision
    f1 = f1_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    precision = precision_score(true_labels, predictions, average='weighted')

    f1_scores.append(f1)
    recall_values.append(recall)
    precision_values.append(precision)

# print results
print(f"\nAverage Recall: {sum(recall_values) / len(recall_values):.2%}")
print(f"Average Precision: {sum(precision_values) / len(precision_values):.2%}")
print(f"Average F1 Score: {sum(f1_scores) / len(f1_scores):.2%}\n")
print(f"Standard Deviation of recalls: {stdev(recall_values):.2f}")
print(f"Standard Deviation of precisions: {stdev(precision_values):.2f}")
print(f"Standard Deviation of f1-scores: {stdev(f1_scores):.2f}\n")
