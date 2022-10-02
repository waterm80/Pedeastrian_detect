import torch.nn as nn

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # input_shape=(3,224,224)
        self.cnn1 = nn.Conv2d(3, 16, kernel_size=5, stride=1) 
        self.relu1 = nn.ReLU(inplace=True) 
        # input_shape=(3,220,220)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # input_shape=(16,110,110)
        self.cnn2 = nn.Conv2d(16,8, kernel_size=11, stride=1) 
        self.relu2 = nn.ReLU(inplace=True) 
        # input_shape=(8,100,100)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # input_shape=(8,50,50)
        self.fc = nn.Linear(8 * 50 * 50, 2)     

    def forward(self, x):
        out = self.cnn1(x) 
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1) 
        out = self.fc(out) 
        return out