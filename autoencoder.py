import torch
import torch.nn as nn

class DeepAutoencoder(nn.Module):
    def __init__(self, input_shape):
        super(DeepAutoencoder, self).__init__()
        self.input_shape = input_shape
        self.flatten_size = int(torch.prod(torch.tensor(input_shape)))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() 

        self.linear1 = nn.Linear(self.flatten_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)

        self.linearA = nn.Linear(64, 128)
        self.linearB = nn.Linear(128 + 128, 256) #Output of LinearA and 3
        self.linearC = nn.Linear(256 + 256, 512) # Output of LinearB and 2
        self.linearD = nn.Linear(512 + 512, self.flatten_size) # Output of LinearC and 1

        input_channels = input_shape[0]
        self.conv = torch.nn.Conv2d(input_channels * 2, input_channels, 3, padding="same")
        #self.linearE = nn.Linear(self.flatten_size + self.flatten_size, self.flatten_size) #Output of LinearD and original input

    def forward(self, x):
        x0 = x.view(x.size(0), -1)  # Flatten the image
        x1 = self.relu(self.linear1(x0))
        x2 = self.relu(self.linear2(x1))
        x3 = self.relu(self.linear3(x2))
        x4 = self.relu(self.linear4(x3))

        xA = self.relu(self.linearA(x4))
        xB = self.relu(self.linearB(torch.concat([xA, x3], axis=-1)))
        xC = self.relu(self.linearC(torch.concat([xB, x2], axis=-1)))
        xD = self.relu(self.linearD(torch.concat([xC, x1], axis=-1)))
        xConv_in = torch.concat([xD.reshape(x.shape), x], axis=1)
        xConv_out = self.sigmoid(self.conv(xConv_in))
        return xConv_out
