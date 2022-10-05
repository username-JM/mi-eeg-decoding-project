import torch
import torch.nn as nn


class BCI2021(nn.Module):
    def __init__(self, args, shape):
        super(BCI2021, self).__init__()
        # Check input dimensions
        assert len(shape) == 5, "Dimensions of input shape should be 6. Please check your input data."
        # Declare variables and instance variables.
        cfg = args.cfg
        self.n_trial, self.n_seg, self.n_band, self.chans, self.times = shape
        # CNN
        self.cnn = nn.ModuleList()
        for _ in range(self.n_band):
            self.cnn.append(CNN(cfg['CNN']))
        # Sub-band attention
        if self.n_band != 1:
            self.sub_band_att = Attention(cfg['SubBandAtt'])
        # Attention-based Bi-LSTM
        self.lstm = nn.LSTM(input_size=cfg['LSTM']['input_size'],
                            hidden_size=cfg['LSTM']['hidden_size'],
                            num_layers=cfg['LSTM']['num_layers'],
                            bidirectional=True, batch_first=True)
        self.segment_att = Attention(cfg['SegmentAtt'])
        # FC
        self.fc = nn.Linear(cfg['SegmentAtt'][0], cfg['n_class'])

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = torch.fmod(torch.randn(m.weight.data.shape), 0.01)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, std=0.01)
                if m.bias is not None:
                    nn.init.normal_(m.bias.data, std=0.1)

        # Weight initialization(he)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight.data)
        #         m.bias.data.fill_(0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight.data)
        #         if m.bias is not None:
        #             m.bias.data.fill_(0)

    def forward(self, X):
        # CNN
        cnn_output = []
        for band in range(self.n_band):
            X_band = X[:, :, band, ...]
            X_band = X_band.view(-1, 1, *list(X_band.shape[2:]))
            cnn_output.append(self.cnn[band](X_band))
        # Sub-band attention
        out = torch.cat(cnn_output, dim=1)
        if self.n_band != 1:
            out, _ = self.sub_band_att(out)
        # Attention-based Bi-LSTM
        out = out.view([self.n_trial, self.n_seg, -1]).squeeze()
        out, _ = self.lstm(out)
        out, _ = self.segment_att(out)
        # FC
        out = self.fc(out)
        return out

    def attention_score(self, X):
        out_cnn_array = []
        for i in range(X.size(2)):
            X_cnn = X[:, :, i, ...].clone().detach()
            X_cnn = X_cnn.view([-1, 1, X_cnn.shape[2], X_cnn.shape[3]])
            out_cnn = self.cnn[i](X_cnn)
            out_cnn_array.append(out_cnn)
        out = torch.cat(out_cnn_array, dim=1)
        _, score = self.sub_band_att(out)
        score = score.data.cpu().numpy().mean(axis=0).flatten()
        return score


class CNN(nn.Module):
    def __init__(self, cfg):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(cfg['in_channels'], cfg['out_channels'], (cfg['c_kernel_size'][0], cfg['c_kernel_size'][1]),
                      cfg['c_stride']),
            nn.BatchNorm2d(cfg['out_channels']),
            nn.ELU(),
            nn.MaxPool2d((cfg['p_kernel_size'][0], cfg['p_kernel_size'][1]), cfg['p_stride']),
            nn.Conv2d(cfg['out_channels'], 1, 1),
            nn.ELU()
        )

    def forward(self, X):
        out = self.cnn(X)
        return out


class Attention(nn.Module):
    def __init__(self, cfg):
        super(Attention, self).__init__()
        layers = [nn.Linear(cfg[0], cfg[1]),
                  nn.Tanh(),
                  nn.Linear(cfg[1], 1, bias=False),
                  nn.Softmax(dim=1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        score = out
        out = out * x  # element-wise product
        out = torch.sum(out, axis=1)
        return out, score

