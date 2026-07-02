import os
import csv
import torch
from torch import nn 
from torch.utils.data import DataLoader 
from torchvision import datasets 
from torchvision.transforms import ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomRotation, Compose 
from ninformer import NiNformer

transform = Compose([
 
ToTensor(),
Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])

training_data = datasets.CIFAR10(
                                       root='data',
                                       train=True,
                                       download=True,
                                       transform=transform 
                                       )

test_data = datasets.CIFAR10(
                                       root='data',
                                       train=False,
                                       download=True,
                                       transform=transform 
                                       )                                       

                                     
batch_size = 128

train_dataloader = DataLoader(training_data, batch_size=batch_size,shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


for X, y in test_dataloader:
    print(f"Shape of X [N,C,H,W]:{X.shape}")
    print(f"Shape of y:{y.shape}{y.dtype}")
    break


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device} device") 


class NiNformerImageClassification(NiNformer):
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        d_model=256,
        d_ffn=512,
        seq_len=64,
        num_layers=4,
        dropout=0.5
    ):
        super().__init__(d_model, d_ffn, seq_len, num_layers,dropout)
        self.patcher = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        patches = self.patcher(x)
        batch_size, num_channels, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_channels)
        embedding = self.model(patches)
        embedding = embedding.mean(dim=1) # global average pooling
        out = self.classifier(embedding)
        return out

model = NiNformerImageClassification().to(device)
print(model)



loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)




def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    correct = 0
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
       
       
        pred = model(X)
        loss = loss_fn(pred,y)
        
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, labels = torch.max(pred.data, 1)
        correct += labels.eq(y.data).type(torch.float).sum()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}   [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    train_accuracy = 100. * correct.item() / size
    print(train_accuracy)
    return train_loss,train_accuracy 




def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)            
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")  
    test_accuracy = 100*correct      
    return test_loss, test_accuracy





logname = "/PATH/NiNformer/Experiments_cifar10/logs_ninformer/logs_cifar10.csv"
if not os.path.exists(logname):
  with open(logname, 'w') as logfile:
    logwriter = csv.writer(logfile, delimiter=',')
    logwriter.writerow(['epoch', 'train loss', 'train acc',
                        'test loss', 'test acc'])


epochs = 100
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-----------------------------------")
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    test_loss, test_acc = test(test_dataloader, model, loss_fn)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch+1, train_loss, train_acc,
                            test_loss, test_acc])
print("Done!")


path = "/PATH/NiNformer/Experiments_cifar10/weights_ninformer"
model_name = "NiNformerImageClassification_cifar10"
torch.save(model.state_dict(), f"{path}/{model_name}.pth")
print(f"Saved Model State to {path}/{model_name}.pth ")

