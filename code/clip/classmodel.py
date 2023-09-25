import torch.nn as nn

class ClassModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClassModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim).cuda()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim).cuda()
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        # self.sigmoid = nn.Softmax()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.sigmoid(out)
        out = self.relu(out)
        return out