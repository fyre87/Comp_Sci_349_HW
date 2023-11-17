import sys
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

#from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelBinarizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

import datetime



# Define the One-hot Encoder
def encode(y):
    # One hot encode y
    label_mapping = {'Bad': 0, 'Neutral': 1, 'Good': 2}
    numerical_labels = np.array([label_mapping[label] for label in y])
    numerical_labels = numerical_labels.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    y_transformed = onehot_encoder.fit_transform(numerical_labels)
    return y_transformed

def preprocess_three(filename):
    df = pd.read_csv(filename)
    X = df.drop('Label', axis=1).values
    y = df['Label'].values
    sc = MinMaxScaler()
    X  = sc.fit_transform(X)
    y = encode(y)

    return X,y


def preprocess_mnist(filename):
    df = pd.read_csv(filename, header=None)
    df = df.drop(785, axis=1)
    X = df.drop(0, axis=1).values
    y = df[0].values
    X = np.where(X <=128, 0, 1)

    return X,y

class threeDataset(torch.utils.data.Dataset):
    def __init__(self, file):
        self.X, self.y= preprocess_three(file)
        self.len = self.X.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        X = torch.tensor(self.X,dtype=torch.float32)
        y = torch.tensor(self.y,dtype=torch.float32)
        return X[idx], y[idx]

class mnistDataset(torch.utils.data.Dataset):
    def __init__(self, file):
        self.X,self.y= preprocess_mnist(file)
        self.len = self.X.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        X = torch.tensor(self.X,dtype=torch.float32)
        y = torch.tensor(self.y,dtype=torch.float32)
        return X[idx], y[idx]

class softmax(nn.Module):
    def __init__(self):
        super(softmax, self).__init__()

    def forward(self, x):
        return torch.exp(x)/torch.sum(torch.exp(x), dim=1)

def loadData_three(filename, batch_size):
    D = threeDataset(filename)
    Dload = DataLoader(D, batch_size=batch_size, shuffle=True)
    return Dload

def loadData_mnist(filename, batch_size):
    D = mnistDataset(filename)
    Dload = DataLoader(D, batch_size=batch_size, shuffle=True)
    return Dload

def pltLoss(train_loss,val_loss):
    plt_trainLossX = [i for i in range(len(train_loss))]
    plt_trainLossY = torch.tensor(train_loss).mean(axis=1)
    plt_valLossX = [i for i in range(len(val_loss))]
    plt_valLossY = val_loss
    plt.plot(plt_trainLossX, plt_trainLossY, label='train_loss')
    plt.plot(plt_valLossX, plt_valLossY, label='val_loss')
    plt.legend()
    plt.show()

class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(3, 2)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(2, 3)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        return x

class FeedForward_mnist(nn.Module):
    def __init__(self):
        super(FeedForward_mnist, self).__init__()
        self.linear1 = nn.Linear(28*28, 256)

        self.relu1 = nn.LeakyReLU()
        self.linear2 = nn.Linear(256, 64)
        self.relu2 = nn.LeakyReLU()
        self.linear_out = nn.Linear(64, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear_out(x)
        return x

def train(dataloader, model, loss_func, optimizer):
    model.train()
    train_loss = []

    now = datetime.datetime.now()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        y_hat = torch.softmax(pred.reshape(-1,3), dim=1)
        loss = loss_func(y_hat, y)

        
        loss.backward() #Updates Gradients
        optimizer.step() #changes gradients
        optimizer.zero_grad() #Zeroes out gradients
        

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            iters = 100 * len(X)
            then = datetime.datetime.now()
            iters /= (then - now).total_seconds()
            print(f"loss: {loss:>6f} [{current:>5d}/{17000}] ({iters:.1f} its/sec)")
            now = then
            train_loss.append(loss)
    return train_loss

def test(dataloader, model, loss_func):
    num_batches = 0
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:

            X, y = X.to(device), y.to(device)

            pred = model(X)
            y_hat = torch.softmax(pred.reshape(-1,3), dim=1)

            test_loss += loss_func(y_hat, y)

            num_batches = num_batches + 1
    test_loss /= num_batches

    print(f"Avg Loss: {test_loss:>8f}\n")
    return test_loss

def train_mnist(dataloader, model, loss_func, optimizer):
    model.train()
    train_loss = []

    now = datetime.datetime.now()
    for batch, (X, y) in enumerate(dataloader):
        # print('pixels:', X, 'targets:', y)
        X, y = X.to(device), y.to(device)

        # make some predictions and get the error
        X = X.view(-1, 784)
        # print(a.shape)
        pred = model(X)
        # y_hat = torch.softmax(pred, dim=1)
        loss = loss_func(pred, y.long().flatten())

        optimizer.zero_grad() #Resetting all the gradients
        loss.backward() #Calculating new gradients
        optimizer.step() #Using those gradients to calculate new weights

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            iters = 10 * len(X)
            then = datetime.datetime.now()
            iters /= (then - now).total_seconds()
            print(f"loss: {loss:>6f} [{current:>5d}/{17000}] ({iters:.1f} its/sec)")
            now = then
            train_loss.append(loss)
    return train_loss

def test_mnist(dataloader, model, loss_func):
    size = len(dataloader)
    num_batches = 0
    model.eval()
    test_loss = 0

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for X, y in dataloader:

            X, y = X.reshape(-1, 784).to(device), y.to(device)
            X = X.view(-1, 784)
            pred = model(X)
            _, predicted = torch.max(pred.data, 1)
            n_samples += y.size(0)
            n_correct += (predicted == y).sum().item()

            test_loss += loss_func(pred, y.long().flatten())
            #.item()

            num_batches = num_batches + 1
    test_loss /= num_batches
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the test images: {acc} %')

    print(f"Avg Loss: {test_loss:>8f}\n")
    return test_loss


def trainit(trainL, valL, model, loss, opt, epochs):
    train_loss = []
    val_loss = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n------------------------------- \n")
        losses = train(trainL, model, loss, opt)
        train_loss.append(losses)
        val_loss.append(test(valL, model, loss))
    return train_loss, val_loss

def trainit_mnist(trainL, valL, model, loss, opt, epochs):
    train_loss = []
    val_loss = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n------------------------------- \n")
        losses = train_mnist(trainL, model, loss, opt)
        train_loss.append(losses)
        val_loss.append(test_mnist(valL, model, loss))
    return train_loss, val_loss

class FeedForward_no_bias(nn.Module):
    def __init__(self):
        super(FeedForward_no_bias, self).__init__()

        self.linear1 = nn.Linear(3, 2, bias = False)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(2, 3, bias = False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)

        return x


def train_no_optimizer(dataloader, model, loss_func, learning_rate):
    model.train()
    train_loss = []

    now = datetime.datetime.now()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        y_hat = torch.softmax(pred.reshape(-1,3), dim=1)
        loss = loss_func(y_hat, y)
        loss.backward() # Calculates new gradients

        # Two strategies to update the weights:::
        with torch.no_grad():
            for param in model.parameters():
                param.data -= learning_rate * param.grad

        model.zero_grad() # Makes all the gradients 0

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            iters = 100 * len(X)
            then = datetime.datetime.now()
            iters /= (then - now).total_seconds()
            print(f"loss: {loss:>6f} [{current:>5d}/{17000}] ({iters:.1f} its/sec)")
            now = then
            train_loss.append(loss)
    return train_loss


def trainit_no_optimizer(trainL, valL, model, loss, epochs, learning_rate):
    train_loss = []
    val_loss = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n------------------------------- \n")
        losses = train_no_optimizer(trainL, model, loss, learning_rate)
        train_loss.append(losses)
        val_loss.append(test(valL, model, loss))
    return train_loss, val_loss

#needs update
def testIt(testD, model):
    model.eval()
    predictedLabels = []
    trueLabels = []
    with torch.no_grad():
        for ex in testD:
            x, y = ex
            x, y = x.reshape(-1,3).to(device), y.to(device)
            pred = model(x)
            y = torch.argmax(y.data)
            _, predicted = torch.max(pred.data, 1)
            predictedLabels.append(int(predicted))
            trueLabels.append(int(y))


    cm = confusion_matrix(y_true = trueLabels, y_pred = predictedLabels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)


    f1 = f1_score(trueLabels, predictedLabels, average=None)
    hm = classification_report(trueLabels, predictedLabels, digits=2)
    return cm, f1, hm

def testIt_mnist(testD, model):
    model.eval()
    predictedLabels = []
    trueLabels = []
    with torch.no_grad():
        for ex in testD:
            x, y = ex
            x, y = x.reshape(-1, 784).to(device), y.to(device)
            pred = model(x)

            _, predicted = torch.max(pred.data, 1)
            predictedLabels.append(int(predicted))
            trueLabels.append(int(y))


    cm = confusion_matrix(y_true = trueLabels, y_pred = predictedLabels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)


    f1 = f1_score(trueLabels, predictedLabels, average=None)

    hm = classification_report(trueLabels, predictedLabels, digits=2)
    return cm, f1, hm

def classify_insurability():
    
    ## FF, no regularization
    ff = FeedForward().to(device)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(ff.parameters(), lr=1e-1)

    epochs = 11
    train_loader = loadData_three('three_train.csv', 1)
    val_loader = loadData_three('three_valid.csv', 1)

    for images, labels in train_loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        break


    train_loss, val_loss = trainit(train_loader, val_loader, ff, loss_func, optimizer, epochs)
    pltLoss(train_loss, val_loss)

    #final evaluation
    testD = threeDataset('three_test.csv')
    cm, f1, hm = testIt(testD, ff)
    print('confusion matrix:\n', cm)
    print('f1:\n', f1)
    print('all info:\n', hm)
    
    # insert code to train simple FFNN and produce evaluation metrics
    
def classify_mnist():
    
    ## FF, no regularization
    ff = FeedForward_mnist().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ff.parameters(), lr=1e-3)

    epochs = 5
    train_loader = loadData_mnist('mnist_train.csv', 64)
    val_loader = loadData_mnist('mnist_valid.csv', 64)

    #training w/ train & val dataset
    train_loss, val_loss = trainit_mnist(train_loader, val_loader, ff, loss_func, optimizer, epochs)
    pltLoss(train_loss, val_loss)

    #final evaluation
    testD = mnistDataset('mnist_test.csv')
    cm, f1, hm = testIt_mnist(testD, ff)
    print('confusion matrix:\n', cm)
    print('f1:\n', f1)
    print('all info:\n', hm)
    
def classify_mnist_reg():
    
    ## FF, with regularization
    L2ff = FeedForward_mnist().to(device)
    loss_func = nn.CrossEntropyLoss()

    #adding regularization with weight_decay variable
    L2optimizer = torch.optim.Adam(L2ff.parameters(), lr=1e-3, weight_decay=1e-5)

    epochs = 6
    train_loader = loadData_mnist('mnist_train.csv', 64)
    val_loader = loadData_mnist('mnist_valid.csv', 64)

    L2train_loss, L2val_loss = trainit_mnist(train_loader, val_loader, L2ff, loss_func, L2optimizer, epochs)
    pltLoss(L2train_loss, L2val_loss)

    testD = mnistDataset('mnist_test.csv')
    cm, f1, hm = testIt_mnist(testD, L2ff)
    print('confusion matrix:\n', cm)
    print('f1:\n', f1)
    print('all info:\n', hm)
    
def classify_insurability_manual():
    ## FF, no regularization
    ff = FeedForward_no_bias().to(device)
    loss_func = nn.MSELoss()

    epochs = 16
    learning_rate = 1e-1
    train_loader = loadData_three('three_train.csv', 1)
    val_loader = loadData_three('three_valid.csv', 1)

    for images, labels in train_loader:
        print('Image batch dimensions:', images.shape) #NCHW
        print('Image label dimensions:', labels.shape)
        break


    train_loss, val_loss = trainit_no_optimizer(train_loader, val_loader, ff, loss_func, epochs, learning_rate)
    pltLoss(train_loss, val_loss)

    #final evaluation
    testD = threeDataset('three_test.csv')
    cm, f1, hm = testIt(testD, ff)
    print('confusion matrix:\n', cm)
    print('f1:\n', f1)
    print('all info:\n', hm)

    
    
def main():
    classify_insurability()
    classify_mnist()
    classify_mnist_reg()
    classify_insurability_manual()
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main()
