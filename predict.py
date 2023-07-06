import pandas as pd
from dataset import HW3Dataset
import torch_geometric
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = SAGEConv(dataset.num_features, 128, aggr="mean")
        self.conv2 = SAGEConv(128, 256, aggr="mean")
        self.conv3 = SAGEConv(256, 512, aggr="mean")
        self.conv4 = SAGEConv(512, 1024, aggr="mean")
        self.conv5 = SAGEConv(1024, dataset.num_classes, aggr="mean")

        self.dropout = torch.nn.Dropout(0.5)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.bn3 = torch.nn.BatchNorm1d(512)
        self.bn4 = torch.nn.BatchNorm1d(1024)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.bn1(x)

        x = self.conv2(x, data.edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.bn2(x)

        x = self.conv3(x, data.edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.bn3(x)

        x = self.conv4(x, data.edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.bn4(x)

        x = self.conv5(x, data.edge_index)
        return F.log_softmax(x, dim=1)


dataset = HW3Dataset(root='data/hw3/')
data = dataset[0]
data.y = data.y.flatten()
size = data.train_mask.shape[0] + data.val_mask.shape[0]
max_index = data.train_mask.ravel()[-1].item()
data.train_mask = torch.as_tensor([True if x <= max_index else False for x in range(size)])
data.val_mask = torch.as_tensor([True if x > max_index else False for x in range(size)])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
model.load_state_dict(torch.load('model.pt', map_location=torch.device(device)))
model.eval()

logit = model.forward(data)[data.train_mask]
pred = torch.argmax(logit, dim=1)
df = pd.DataFrame()
df['idx'] = [*range(80000)]
df['prediction'] = pred.cpu().numpy()
df.to_csv('prediction.csv', index=False)
