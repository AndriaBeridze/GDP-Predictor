import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as ts
from sklearn.preprocessing import MinMaxScaler as mms
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Set seed
np.random.seed(0)

# Load data
path = 'cleaned_GDP.csv'
df = pd.read_csv(path, decimal = ',')

for col in df.columns.values:
    df[col] = df[col].astype(float)

target = df['GDP ($ per capita)']
features = df.loc[:, df.columns != 'GDP ($ per capita)']

# Scale Data
scaler = mms()
if len(features.shape) == 1:
    features = features.reshape(-1,1)
features_scaled = scaler.fit_transform(features)

# Split and Reshape Data
x_train, x_test, y_train, y_test = ts(features, target, test_size=0.2, random_state=43)

y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)

x_train = np.array(x_train)
x_test = np.array(x_test)

# Make arrays into tensors
x_train = torch.tensor(x_train.astype(float), dtype=torch.float32)
x_test = torch.tensor(x_test.astype(float), dtype=torch.float32)
y_train = torch.tensor(y_train.astype(float), dtype=torch.float32)
y_test = torch.tensor(y_test.astype(float), dtype=torch.float32)


# initalize model architecture, loss function, and optimizer

# tune model hidden layers
l1 = 20
l2 = 10
model = nn.Sequential(
            nn.Linear(29, l1), # takes in 29 features (the input layer) and outputs "l1"
            nn.ReLU(),
            nn.Linear(l1, l2), # takes in "l1" features, outputs "l2"
            nn.ReLU(),
            nn.Linear(l2, 1), # takes in "l2" features, outputs one feature
        )

loss_criterion = nn.L1Loss() # l1 loss is getting better results compared to MSE

# choose optimizer function from torch.optim socs
# optimizer = optim.LBFGS(model.parameters(), lr = 0.0001) # customizable
optimizer = optim.SGD(model.parameters(), lr = 0.001, weight_decay=0.0001) # customizable
# optimizer = optim.Adamax(model.parameters(), lr=0.0001, weight_decay=0.0001)

# choose scheduler function
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# training
epochs = 1000

patience = 10 #number of epochs to wait for improvement
best_loss_val = float("inf")
epochs_since_improvement = 0

def closure():
    y_pred = model(x_train)
    loss = loss_criterion(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    return loss

for epoch in range(epochs):
    model.train() # sets the model into training mode

    loss = optimizer.step(closure)
    scheduler.step(loss)
    #early stopping --- early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max")
    if loss.item() < best_loss_val: 
        best_loss_val = loss 
        epochs_since_improvement = 0
        #save the best model weights
        best_model_weights = model.state_dict()
    else:
        epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print(f"Early stopping at {epoch}.")
            break
    
    if ((epoch + 1) % 50 == 0):
        print(f'Epoch: {epoch} and loss: {loss.item(): 4f}')

model.load_state_dict(best_model_weights)   

# predictions/testing

with torch.no_grad():
    model.eval() # sets model into test/validation mode
    y_hat = model(x_test)
    
# y_hat = y_hat.numpy()

y_hat = y_hat.numpy().flatten()
y_test = np.array(y_test).flatten()

# evaluating the model
r2 = r2_score(y_test, y_hat)
mae = mean_absolute_error(y_test,y_hat)

print("R2 Score:",r2)
print("MAE Score:",mae)

