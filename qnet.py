from torch import nn

class QNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(QNet, self).__init__()

        assert(len(hidden_dims) > 0)

        self.net = nn.Sequential()
        self.net.add_module(f"hidden_0", nn.Sequential(nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()))
        for i in range(len(hidden_dims)-1):
            self.net.add_module(f"hidden_{i+1}", nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()))
        self.net.add_module("output", nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, observation):
        return self.net(observation)