import torch
import csv
import os as os
from torch.utils.data import Dataset, DataLoader
import onnx
from model import MLPs
from torch import nn


# device
device = torch.device("cuda:0")
print(device)

# 超参
learning_rate = 0.0005
hidden_d = 100000000007
num_epochs = 300
train_batch_size = 100
test_batch_size = 100
num_works = 2

# 数据集
string_labels = []
int_labels = []

class FDataset(Dataset):
    def __init__(self, folder_path):
        self.csv_files = os.listdir(folder_path)
        self.data = []
        self.labels = []
        self.idx = 0
        for csv_file in self.csv_files:
            label = csv_file.split('.')[0]
            string_labels.append(label)
            int_labels.append(self.idx)
            self.idx += 1
            with open(os.path.join(folder_path, csv_file), 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    row = row[0:3]
                    data = torch.tensor([float(x) for x in row])
                    #print(label, data)
                    self.data.append(data)
                    self.labels.append(string_labels.index(label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

fdataset = FDataset('Dataset_csv')
train_dataset, test_dataset = torch.utils.data.random_split(
    fdataset, [int(len(fdataset) * 0.8), len(fdataset) - int(len(fdataset) * 0.8)]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    sampler=None,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=True,
    sampler=None,
    num_workers=0,
)


# 将模型保存为onnx
def save_model(model):
    dummy_input = torch.randn(torch.Size([3]))
    torch.onnx.export(model, dummy_input, "mlp.onnx")

    onnx_model = onnx.load("mlp.onnx")
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))




# 训练
model = MLPs()

loss_f = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
save_model_name = 'mlp_without_dropout.pt'

for epoch in range(num_epochs):
    for batch, (x, y) in enumerate(train_loader):
        x.to(device)
        y.to(device)

        y_pred = model(x)
        loss = loss_f(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch}, Loss" {loss.item()}')

    torch.save(model, save_model_name)
    save_model(model)

    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in test_loader:
            y_pred = model(x)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        #print(f'predicted: {predicted}\n')
        print(f'Epoch: {epoch}, Accuracy: {100 * correct / total}%')



