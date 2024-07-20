import torch
from torch import nn

class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        self.gap = nn.AvgPool2d(int(WH / stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

        self.conv=nn.Conv2d(2*features,features,kernel_size=3,padding=1,stride=1)

    def forward(self, x, y):
        # for i, conv in enumerate(self.convs):
        #     fea = conv(x).unsqueeze_(dim=1)
        #     if i == 0:
        #         feas = fea
        #     else:
        #         feas = torch.cat([feas, fea], dim=1)
        x = x.unsqueeze(dim=1)
        y = y.unsqueeze(dim=1)
        feas = torch.cat([x, y], dim=1)

        fea_U = torch.sum(feas, dim=1)

        fea_s = self.gap(fea_U).squeeze().unsqueeze(dim=0)

        # fea_s = fea_s.unsqueeze(1)
        # print(fea_s.shape)
        fea_z = self.fc(fea_s)


        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)


        fea_v = (feas * attention_vectors)

        fea_v = fea_v.sum(dim=1)

        return fea_v


if __name__ == '__main__':
    x = torch.rand(56, 256, 56, 56)
    conv = SKConv(256, 56, 3, 8, 2)
    out = conv(x)
    criterion = nn.L1Loss()
    loss = criterion(out, x)
    loss.backward()
    print('out shape : {}'.format(out.shape))
    print('loss value : {}'.format(loss))