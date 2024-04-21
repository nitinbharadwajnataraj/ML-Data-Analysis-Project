import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from main import df_fitting_and_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# Define neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def prepare_NN_df():
    df, style_df = df_fitting_and_evaluation()
    df.drop(columns=['ID', 'Evaluation'], inplace=True)

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Fit and transform the 'fitting_group' column
    df['fitting_group_encoded'] = label_encoder.fit_transform(df['fitting_group'])

    # Mapping of original values to encoded values
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Label mapping:", label_mapping)

    # Replace 'fitting_group' with 'fitting_group_encoded'
    df.drop(columns=['fitting_group'], inplace=True)
    df.rename(columns={'fitting_group_encoded': 'fitting_group'}, inplace=True)
    df.drop(columns=['Evaluation', 'ID', 'fitting_distance'], inplace=True)

    return df


def train_NN():
    df = prepare_NN_df()
    X = df.iloc[:, 0:4].values
    y = df.iloc[:, 4].values

    # Split data into train, validation, and test sets
    x_main, x_test, y_main, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_main, y_main, test_size=0.2, stratify=y_main)

    input_dim = X.shape[1]
    hidden_dim = 32
    output_dim = len(pd.unique(df['fitting_group']))

    # Initialize neural network
    model = NeuralNetwork(input_dim, hidden_dim, output_dim)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert data to PyTorch tensors
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.LongTensor(y_train)
    x_val_tensor = torch.FloatTensor(x_val)
    y_val_tensor = torch.LongTensor(y_val)

    # Training the model
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Validate the model
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(torch.FloatTensor(x_test))
        _, predicted = torch.max(outputs, 1)

        # Print confusion matrix and classification report
        print("Confusion Matrix for TEST:")
        print(confusion_matrix(y_test, predicted.numpy()))
        print("Classification Report for TEST:")
        print(classification_report(y_test, predicted.numpy()))


if __name__ == "__main__":
    train_NN()
