import torch
from torch import nn


VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):

    def __init__(self,in_channels,num_classes,architecture_type):
        super().__init__()

        self.in_channels = in_channels
        self.conv_layers = self.create_conv(VGG_types[architecture_type])
        # self.pool = nn.MaxPool2d(kernel_size=(2,2))

        self.classifier  = nn.Sequential(
                        nn.Linear(512*7*7,4096),
                        nn.ReLU(),
                        nn.Linear(4096,4096),
                        nn.ReLU(),
                        nn.Linear(4096,num_classes),
                        # nn.Softmax(dim = 1) #uncomment if you want softmax prob
        )


    def forward(self,x):

        batch_size = x.shape[0]
        x = self.conv_layers(x)
        x = x.view(batch_size,-1)
        out = self.classifier(x)

        return out

    def create_conv(self,architecture_list):

        conv_layers = []
        in_channel = self.in_channels
        for x in architecture_list:
            if type(x) == int:
                conv_layers += [nn.Conv2d(in_channels=in_channel,out_channels=x,kernel_size=(3,3),
                                stride=1,padding=1),
                                nn.ReLU()]
                
                in_channel = x
            elif x == "M":
                conv_layers += [nn.MaxPool2d(kernel_size=(2,2),stride = 2)]
        
        return nn.Sequential(*conv_layers)
                
if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    model = VGG(in_channels=3,num_classes=10,architecture_type='VGG16')
    
    model.to(device)

    x = torch.randn(3,3,224,224).to(device)
    # print(model)
    out = model(x)
    print(out.shape)