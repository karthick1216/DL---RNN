# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset



## DESIGN STEPS
### STEP 1: Load and normalize data, create sequences.

### STEP 2: Convert data to tensors and set up DataLoader.

### STEP 3: Define the RNN model architecture.

### STEP 4: Summarize, compile with loss and optimizer.

### STEP 5: Train the model with loss tracking.

### STEP 6: Predict on test data, plot actual vs. predicted prices.

## PROGRAM

### Name: KARTHICK S

### Register Number: 212224230114

```python
# Define RNN Model
class RNNModel(nn.Module):
  def __init__(self,input_size=1,hidden_size=64,num_layers=2,output_size=1):
    super(RNNModel,self).__init__()
    self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
    self.fc = nn.Linear(hidden_size,output_size)
  def forward(self,x):
    out,_=self.rnn(x)
    out=self.fc(out[:,-1,:])
    return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torchinfo import summary

# input_size = (batch_size, seq_len, input_size)
summary(model, input_size=(64, 60, 1))

#-----hard code-----
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

# Train the Model
## Step 3: Train the Model
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    train_losses = []
    model.train()
    for epoch in range(epochs):
      total_loss = 0
      for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
      train_losses.append(total_loss / len(train_loader))
      print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')
    # Plot training loss
    print('Name: karthick s')
    print('Register Number: 212224230114')
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()
train_model(model,train_loader,criterion,optimizer)

```

### OUTPUT
<img width="365" height="49" alt="image" src="https://github.com/user-attachments/assets/a0e275f1-2a59-4e47-a6b1-085d40b9e111" />
<img width="1220" height="821" alt="image" src="https://github.com/user-attachments/assets/d678472a-90e4-4aa2-bb9b-8c78e42c7c6d" />

<img width="365" height="49" alt="image" src="https://github.com/user-attachments/assets/6a8b58de-ca1c-4d69-9612-fd2a3cc3c16e" />
<img width="1100" height="747" alt="image" src="https://github.com/user-attachments/assets/4f729b4e-3690-4a8e-881a-9402aca6b0db" />
<img width="784" height="78" alt="image" src="https://github.com/user-attachments/assets/39d59be8-2fc2-4e91-b2d8-58b44f36a7b8" />



## RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.
