import torch
import torch.optim
import torch.utils.data
import pandas as pd
import h5py

test_filename = "./data/test_data.h5"
train_filename = "./data/train_data.h5"
class BicycleDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        super().__init__()
        self.h5_file = h5py.File(file_path, 'r')
    
    def __getitem__(self, index): 
        s = self.h5_file["data"]["table"][index]
        return (torch.Tensor(s[2][:-1]), torch.Tensor([s[1][0], s[2][-1]]))

    def __len__(self):
        return len(self.h5_file["data"]["table"])
        
training_data = BicycleDataset(train_filename)
test_data = BicycleDataset(test_filename)
train_loader = torch.utils.data.DataLoader(training_data, batch_size=128)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#GPU 쓰라고 ... dmgb

N, D_in, H1, H2, H3, D_out = 128, 6, 100, 100, 100, 2
train_steps = 500
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H1),
        torch.nn.ReLU(),
        torch.nn.Linear(H1, H2),
        torch.nn.ReLU(),
        torch.nn.Linear(H2, H3),
        torch.nn.ReLU(),
        torch.nn.Linear(H3, D_out),
        ).to(device)

loss_fn = torch.nn.MSELoss(reduction='sum')

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def train(dataloader, model, train_steps, epoch):
    global device
    log_interval = 1000
    for i in range(train_steps):
        model.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            #data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    i, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss))
    torch.save(model.state_dict(), './data/')

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

train(training_data, model, 3, 3)
test(test_data, model, loss_fn)