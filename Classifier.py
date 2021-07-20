import torch
from torch import nn
from torch import optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split
from PIL import Image
from torchvision.transforms import ToTensor
import sys


batch_size = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = datasets.MNIST(root="./", train=True, transform=transforms.ToTensor(), download=False)
train, val = random_split(train_data, [55000, 5000])

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)

# Defining a model
print("Defining model")
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# Define an Optimiser
print("Defining Optimiser")
optimiser = optim.SGD(model.parameters(), lr=1e-2)

# Define loss
print("Defining Loss")
loss = nn.CrossEntropyLoss()

model = model.to(device)
print("Training Network...")
# Training and validation Loops
nb_epochs = 100
for epoch in range(nb_epochs):

    losses = list()
    for batch in train_loader:

        x, y = batch

        batch_size = x.size()
        x = x.view(batch_size, -1)

        # send computations to GPU
        x = x.to(device)
        y = y.to(device)

        # The fist step is the foward step
        logits = model(x)

        # Compute objective function
        K = loss(logits, y)

        # Clean the gradients
        model.zero_grad()

        # Compute partial derivatives of K wrt parameters
        K.backward()

        # Step in the opposite direction of gradient
        optimiser.step()

        # save loss to the losses list()
        losses.append(K.item())

    print(f'Epoch {epoch+1}, Cross-Entropy training loss: {torch.tensor(losses).mean():.2f}')

    losses = list()
    for batch in val_loader:
        x, y = batch

        batch_size = x.size()
        x = x.view(batch_size, -1)

        x = x.to(device)
        y = y.to(device)

        # The fist step is the foward step
        with torch.no_grad():
            logits = model(x)

        # Compute objective function
        K = loss(logits, y)

        losses.append(K.item())

    print(f'Epoch {epoch + 1}, Cross-Entropy validation loss: {torch.tensor(losses).mean():.2f}')
print("Done!...")

while True:
    print("Please enter a filepath:\n")
    filepath = input()
    if filepath == "exit":
        print("Exiting...")
        sys.exit()
    else:
        the_image = Image.open(filepath)
        img = ToTensor()(the_image).unsqueeze(0)
        img = img.to(device)

        with torch.no_grad():
            log_ps = model(img)

            # Output of the network are log-probabilities, need to take exponential for probabilities
            ps = torch.exp(log_ps).cpu()
            probability = list(ps.numpy()[0])
            predicted_label = probability.index(max(probability))

        print(f'Classifier:{predicted_label}\n')






