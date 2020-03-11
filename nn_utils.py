import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import datetime
import time
import pandas as pd


class AE_basic(nn.Module):
    def __init__(self, nodes, no_last_bias=False):
        super(AE_basic, self).__init__()
        n_layers = len(nodes)
        ins_n_outs = []
        en_modulelist = nn.ModuleList()
        de_modulelist = nn.ModuleList()
        for ii in range(n_layers // 2):
            ins = nodes[ii]
            outs = nodes[ii + 1]
            ins_n_outs.append((ins, outs))
            en_modulelist.append(nn.Linear(ins, outs))
            en_modulelist.append(nn.Tanh())
        for ii in range(n_layers // 2):
            ii += n_layers // 2
            ins = nodes[ii]
            outs = nodes[ii + 1]
            de_modulelist.append(nn.Linear(ins, outs))
            de_modulelist.append(nn.Tanh())

        de_modulelist = de_modulelist[:-1]  # Remove Tanh activation from output layer
        if no_last_bias:
            de_modulelist = de_modulelist[:-1]
            de_modulelist.append(nn.Linear(nodes[-2], nodes[-1], bias=False))

        self.encoder = nn.Sequential(*en_modulelist)

        self.decoder = nn.Sequential(*de_modulelist)

        node_string = ''
        for layer in nodes:
            node_string = node_string + str(layer) + '-'
        node_string = node_string[:-1]
        self.node_string = node_string

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))

    def get_node_string(self):
        return self.node_string


class AE_LeakyReLU(nn.Module):
    def __init__(self, nodes, no_last_bias=False):
        super(AE_LeakyReLU, self).__init__()
        n_layers = len(nodes)
        ins_n_outs = []
        en_modulelist = nn.ModuleList()
        de_modulelist = nn.ModuleList()
        for ii in range(n_layers // 2):
            ins = nodes[ii]
            outs = nodes[ii + 1]
            ins_n_outs.append((ins, outs))
            en_modulelist.append(nn.Linear(ins, outs))
            en_modulelist.append(nn.LeakyReLU())
        for ii in range(n_layers // 2):
            ii += n_layers // 2
            ins = nodes[ii]
            outs = nodes[ii + 1]
            de_modulelist.append(nn.Linear(ins, outs))
            de_modulelist.append(nn.LeakyReLU())

        de_modulelist = de_modulelist[:-1]  # Remove LeakyReLU activation from output layer
        if no_last_bias:
            de_modulelist = de_modulelist[:-1]
            de_modulelist.append(nn.Linear(nodes[-2], nodes[-1], bias=False))

        self.encoder = nn.Sequential(*en_modulelist)

        self.decoder = nn.Sequential(*de_modulelist)

        node_string = ''
        for layer in nodes:
            node_string = node_string + str(layer) + '-'
        node_string = node_string[:-1]
        self.node_string = node_string

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))

    def get_node_string(self):
        return self.node_string


class AE_bn(nn.Module):
    def __init__(self, nodes, no_last_bias=False):
        super(AE_bn, self).__init__()
        n_layers = len(nodes)
        ins_n_outs = []
        en_modulelist = nn.ModuleList()
        de_modulelist = nn.ModuleList()
        for ii in range(n_layers // 2):
            ins = nodes[ii]
            outs = nodes[ii + 1]
            ins_n_outs.append((ins, outs))
            en_modulelist.append(nn.Linear(ins, outs))

            en_modulelist.append(nn.BatchNorm1d(outs))
            en_modulelist.append(nn.Tanh())
        for ii in range(n_layers // 2):
            ii += n_layers // 2
            ins = nodes[ii]
            outs = nodes[ii + 1]
            de_modulelist.append(nn.Linear(ins, outs))

            de_modulelist.append(nn.BatchNorm1d(outs))
            de_modulelist.append(nn.Tanh())

        de_modulelist = de_modulelist[:-2]  # Remove Tanh activation and BatchNorm1d from output layer
        if no_last_bias:
            de_modulelist = de_modulelist[:-1]
            de_modulelist.append(nn.Linear(nodes[-2], nodes[-1], bias=False))

        self.encoder = nn.Sequential(*en_modulelist)
        self.decoder = nn.Sequential(*de_modulelist)

        node_string = ''
        for layer in nodes:
            node_string = node_string + str(layer) + '-'
        node_string = node_string[:-1]
        self.node_string = node_string

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))

    def get_node_string(self):
        return self.node_string


class AE_bn_LeakyReLU(nn.Module):
    def __init__(self, nodes, no_last_bias=False):
        super(AE_bn_LeakyReLU, self).__init__()
        n_layers = len(nodes)
        ins_n_outs = []
        en_modulelist = nn.ModuleList()
        de_modulelist = nn.ModuleList()
        for ii in range(n_layers // 2):
            ins = nodes[ii]
            outs = nodes[ii + 1]
            ins_n_outs.append((ins, outs))
            en_modulelist.append(nn.Linear(ins, outs))
            en_modulelist.append(nn.LeakyReLU())
            en_modulelist.append(nn.BatchNorm1d(outs))
        for ii in range(n_layers // 2):
            ii += n_layers // 2
            ins = nodes[ii]
            outs = nodes[ii + 1]
            de_modulelist.append(nn.Linear(ins, outs))
            de_modulelist.append(nn.LeakyReLU())
            de_modulelist.append(nn.BatchNorm1d(outs))

        de_modulelist = de_modulelist[:-2]  # Remove LeakyReLU activation and BatchNorm1d from output layer
        if no_last_bias:
            de_modulelist = de_modulelist[:-1]
            de_modulelist.append(nn.Linear(nodes[-2], nodes[-1], bias=False))

        self.encoder = nn.Sequential(*en_modulelist)
        self.decoder = nn.Sequential(*de_modulelist)

        node_string = ''
        for layer in nodes:
            node_string = node_string + str(layer) + '-'
        node_string = node_string[:-1]
        self.node_string = node_string

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))

    def get_node_string(self):
        return self.node_string


class AE_3D_100(nn.Module):
    def __init__(self, n_features=4):
        super(AE_3D_100, self).__init__()
        self.en1 = nn.Linear(n_features, 100)
        self.en2 = nn.Linear(100, 100)
        self.en3 = nn.Linear(100, 50)
        self.en4 = nn.Linear(50, 3)
        self.de1 = nn.Linear(3, 50)
        self.de2 = nn.Linear(50, 100)
        self.de3 = nn.Linear(100, 100)
        self.de4 = nn.Linear(100, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-100-100-50-3-50-100-100-out'


class AE_3D_200(nn.Module):
    def __init__(self, n_features=4):
        super(AE_3D_200, self).__init__()
        self.en1 = nn.Linear(n_features, 200)
        self.en2 = nn.Linear(200, 100)
        self.en3 = nn.Linear(100, 50)
        self.en4 = nn.Linear(50, 3)
        self.de1 = nn.Linear(3, 50)
        self.de2 = nn.Linear(50, 100)
        self.de3 = nn.Linear(100, 200)
        self.de4 = nn.Linear(200, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-200-100-50-3-50-100-200-out'


class AE_3D_small(nn.Module):
    def __init__(self, n_features=4):
        super(AE_3D_small, self).__init__()
        self.en1 = nn.Linear(n_features, 3)
        self.de1 = nn.Linear(3, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en1(x)

    def decode(self, x):
        return self.de1(self.tanh(x))

    def forward(self, x):
        return self.decode(self.encode(x))

    def describe(self):
        return 'in-3-out'


class AE_3D_small_v2(nn.Module):
    def __init__(self, n_features=4):
        super(AE_3D_small_v2, self).__init__()
        self.en1 = nn.Linear(n_features, 8)
        self.en2 = nn.Linear(8, 3)
        self.de1 = nn.Linear(3, 8)
        self.de2 = nn.Linear(8, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en2(self.tanh(self.en1(x)))

    def decode(self, x):
        return self.de2(self.tanh(self.de1(self.tanh(x))))

    def forward(self, x):
        return self.decode(self.encode(x))

    def describe(self):
        return 'in-8-3-8-out'


class AE_big(nn.Module):
    def __init__(self, n_features=4):
        super(AE_big, self).__init__()
        self.en1 = nn.Linear(n_features, 8)
        self.en2 = nn.Linear(8, 6)
        self.en3 = nn.Linear(6, 4)
        self.en4 = nn.Linear(4, 3)
        self.de1 = nn.Linear(3, 4)
        self.de2 = nn.Linear(4, 6)
        self.de3 = nn.Linear(6, 8)
        self.de4 = nn.Linear(8, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-8-6-4-3-4-6-8-out'


class AE_big_no_last_bias(nn.Module):
    def __init__(self, n_features=4):
        super(AE_big_no_last_bias, self).__init__()
        self.en1 = nn.Linear(n_features, 8)
        self.en2 = nn.Linear(8, 6)
        self.en3 = nn.Linear(6, 4)
        self.en4 = nn.Linear(4, 3)
        self.de1 = nn.Linear(3, 4)
        self.de2 = nn.Linear(4, 6)
        self.de3 = nn.Linear(6, 8)
        self.de4 = nn.Linear(8, n_features, bias=False)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-8-6-4-3-4-6-8-out'


class AE_3D_50(nn.Module):
    def __init__(self, n_features=4):
        super(AE_3D_50, self).__init__()
        self.en1 = nn.Linear(n_features, 50)
        self.en2 = nn.Linear(50, 50)
        self.en3 = nn.Linear(50, 20)
        self.en4 = nn.Linear(20, 3)
        self.de1 = nn.Linear(3, 20)
        self.de2 = nn.Linear(20, 50)
        self.de3 = nn.Linear(50, 50)
        self.de4 = nn.Linear(50, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-50-50-20-3-20-50-50-out'


class AE_3D_50_no_last_bias(nn.Module):
    def __init__(self, n_features=4):
        super(AE_3D_50_no_last_bias, self).__init__()
        self.en1 = nn.Linear(n_features, 50)
        self.en2 = nn.Linear(50, 50)
        self.en3 = nn.Linear(50, 20)
        self.en4 = nn.Linear(20, 3)
        self.de1 = nn.Linear(3, 20)
        self.de2 = nn.Linear(20, 50)
        self.de3 = nn.Linear(50, 50)
        self.de4 = nn.Linear(50, n_features, bias=False)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-50-50-20-3-20-50-50-out'


class AE_3D_50cone(nn.Module):
    def __init__(self, n_features=4):
        super(AE_3D_50cone, self).__init__()
        self.en1 = nn.Linear(n_features, 50)
        self.en2 = nn.Linear(50, 30)
        self.en3 = nn.Linear(30, 20)
        self.en4 = nn.Linear(20, 3)
        self.de1 = nn.Linear(3, 20)
        self.de2 = nn.Linear(20, 20)
        self.de3 = nn.Linear(20, 20)
        self.de4 = nn.Linear(20, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-50-50-20-3-20-50-50-out'


class AE_3D_50_bn_drop(nn.Module):
    def __init__(self, n_features=4, dropout=0):
        super(AE_3D_50_bn_drop, self).__init__()
        if type(dropout) is list:
            p1 = dropout[0]
            p2 = dropout[1]
            p3 = dropout[2]
            p4 = dropout[3]
            p5 = dropout[4]
            p6 = dropout[5]
            p7 = dropout[6]
        else:
            p1 = dropout
            p2 = dropout
            p3 = dropout
            p4 = dropout
            p5 = dropout
            p6 = dropout
            p7 = dropout
        #self.bn0 = nn.BatchNorm1d(4)
        self.en1 = nn.Linear(n_features, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.en2 = nn.Linear(50, 50)
        self.bn2 = nn.BatchNorm1d(50)
        self.en3 = nn.Linear(50, 20)
        self.bn3 = nn.BatchNorm1d(20)
        self.en4 = nn.Linear(20, 3)
        self.bn5 = nn.BatchNorm1d(3)
        self.de1 = nn.Linear(3, 20)
        self.bn6 = nn.BatchNorm1d(20)
        self.de2 = nn.Linear(20, 50)
        self.bn7 = nn.BatchNorm1d(50)
        self.de3 = nn.Linear(50, 50)
        self.bn8 = nn.BatchNorm1d(50)
        self.de4 = nn.Linear(50, n_features)
        self.tanh = nn.Tanh()
        self.drop1 = nn.Dropout(p1)
        self.drop2 = nn.Dropout(p2)
        self.drop3 = nn.Dropout(p3)
        self.drop4 = nn.Dropout(p4)
        self.drop5 = nn.Dropout(p5)
        self.drop6 = nn.Dropout(p6)
        self.drop7 = nn.Dropout(p7)

    def encode(self, x):
        h1 = self.drop1(self.bn1(self.tanh(self.en1((x)))))
        h2 = self.drop2(self.bn2(self.tanh(self.en2(h1))))
        h3 = self.drop3(self.bn3(self.tanh(self.en3(h2))))
        z = self.en4(h3)
        return z

    def decode(self, x):
        h5 = self.drop5(self.bn6(self.tanh(self.de1(self.drop4(self.bn5(self.tanh(x)))))))
        h6 = self.drop6(self.bn7(self.tanh(self.de2(h5))))
        h7 = self.drop7(self.bn8(self.tanh(self.de3(h6))))
        return self.de4(h7)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-50-50-20-3-20-50-50-out (bn, dropout)'


class AE_3D_50cone_bn_drop(nn.Module):
    def __init__(self, n_features=4, dropout=0):
        super(AE_3D_50cone_bn_drop, self).__init__()
        if type(dropout) is list:
            p1 = dropout[0]
            p2 = dropout[1]
            p3 = dropout[2]
            p4 = dropout[3]
            p5 = dropout[4]
            p6 = dropout[5]
            p7 = dropout[6]
        else:
            p1 = dropout
            p2 = dropout
            p3 = dropout
            p4 = dropout
            p5 = dropout
            p6 = dropout
            p7 = dropout
        #self.bn0 = nn.BatchNorm1d(4)
        self.en1 = nn.Linear(n_features, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.en2 = nn.Linear(50, 30)
        self.bn2 = nn.BatchNorm1d(30)
        self.en3 = nn.Linear(30, 20)
        self.bn3 = nn.BatchNorm1d(20)
        self.en4 = nn.Linear(20, 3)
        self.bn5 = nn.BatchNorm1d(3)
        self.de1 = nn.Linear(3, 20)
        self.bn6 = nn.BatchNorm1d(20)
        self.de2 = nn.Linear(20, 20)
        self.bn7 = nn.BatchNorm1d(20)
        self.de3 = nn.Linear(20, 20)
        self.bn8 = nn.BatchNorm1d(20)
        self.de4 = nn.Linear(20, n_features)
        self.tanh = nn.Tanh()
        self.drop1 = nn.Dropout(p1)
        self.drop2 = nn.Dropout(p2)
        self.drop3 = nn.Dropout(p3)
        self.drop4 = nn.Dropout(p4)
        self.drop5 = nn.Dropout(p5)
        self.drop6 = nn.Dropout(p6)
        self.drop7 = nn.Dropout(p7)

    def encode(self, x):
        h1 = self.drop1(self.bn1(self.tanh(self.en1((x)))))
        h2 = self.drop2(self.bn2(self.tanh(self.en2(h1))))
        h3 = self.drop3(self.bn3(self.tanh(self.en3(h2))))
        z = self.en4(h3)
        return z

    def decode(self, x):
        h5 = self.drop5(self.bn6(self.tanh(self.de1(self.drop4(self.bn5(self.tanh(x)))))))
        h6 = self.drop6(self.bn7(self.tanh(self.de2(h5))))
        h7 = self.drop7(self.bn8(self.tanh(self.de3(h6))))
        return self.de4(h7)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-50-50-20-3-20-50-50-out (bn, dropout)'


class AE_3D_100_bn_drop(nn.Module):
    def __init__(self, n_features=4, dropout=0):
        super(AE_3D_100_bn_drop, self).__init__()
        if type(dropout) is list:
            p1 = dropout[0]
            p2 = dropout[1]
            p3 = dropout[2]
            p4 = dropout[3]
            p5 = dropout[4]
            p6 = dropout[5]
            p7 = dropout[6]
        else:
            p1 = dropout
            p2 = dropout
            p3 = dropout
            p4 = dropout
            p5 = dropout
            p6 = dropout
            p7 = dropout
        self.en1 = nn.Linear(n_features, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.en2 = nn.Linear(100, 80)
        self.bn2 = nn.BatchNorm1d(80)
        self.en3 = nn.Linear(80, 50)
        self.bn3 = nn.BatchNorm1d(50)
        self.en4 = nn.Linear(50, 3)
        self.bn5 = nn.BatchNorm1d(3)
        self.de1 = nn.Linear(3, 50)
        self.bn6 = nn.BatchNorm1d(50)
        self.de2 = nn.Linear(50, 80)
        self.bn7 = nn.BatchNorm1d(80)
        self.de3 = nn.Linear(80, 100)
        self.bn8 = nn.BatchNorm1d(100)
        self.de4 = nn.Linear(100, n_features)
        self.tanh = nn.Tanh()
        self.drop1 = nn.Dropout(p1)
        self.drop2 = nn.Dropout(p2)
        self.drop3 = nn.Dropout(p3)
        self.drop4 = nn.Dropout(p4)
        self.drop5 = nn.Dropout(p5)
        self.drop6 = nn.Dropout(p6)
        self.drop7 = nn.Dropout(p7)

    def encode(self, x):
        h1 = self.drop1(self.bn1(self.tanh(self.en1(x))))
        h2 = self.drop2(self.bn2(self.tanh(self.en2(h1))))
        h3 = self.drop3(self.bn3(self.tanh(self.en3(h2))))
        z = self.en4(h3)
        return z

    def decode(self, x):
        h5 = self.drop5(self.bn6(self.tanh(self.de1(self.drop4(self.bn5(self.tanh(x)))))))
        h6 = self.drop6(self.bn7(self.tanh(self.de2(h5))))
        h7 = self.drop7(self.bn8(self.tanh(self.de3(h6))))
        return self.de4(h7)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        pass


class AE_3D_100cone_bn_drop(nn.Module):
    def __init__(self, n_features=4, dropout=0):
        super(AE_3D_100cone_bn_drop, self).__init__()
        if type(dropout) is list:
            p1 = dropout[0]
            p2 = dropout[1]
            p3 = dropout[2]
            p4 = dropout[3]
            p5 = dropout[4]
            p6 = dropout[5]
            p7 = dropout[6]
        else:
            p1 = dropout
            p2 = dropout
            p3 = dropout
            p4 = dropout
            p5 = dropout
            p6 = dropout
            p7 = dropout
        self.en1 = nn.Linear(n_features, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.en2 = nn.Linear(100, 80)
        self.bn2 = nn.BatchNorm1d(80)
        self.en3 = nn.Linear(80, 40)
        self.bn3 = nn.BatchNorm1d(40)
        self.en4 = nn.Linear(40, 3)
        self.bn5 = nn.BatchNorm1d(3)
        self.de1 = nn.Linear(3, 50)
        self.bn6 = nn.BatchNorm1d(50)
        self.de2 = nn.Linear(50, 50)
        self.bn7 = nn.BatchNorm1d(50)
        self.de3 = nn.Linear(50, 20)
        self.bn8 = nn.BatchNorm1d(20)
        self.de4 = nn.Linear(20, n_features)
        self.tanh = nn.Tanh()
        self.drop1 = nn.Dropout(p1)
        self.drop2 = nn.Dropout(p2)
        self.drop3 = nn.Dropout(p3)
        self.drop4 = nn.Dropout(p4)
        self.drop5 = nn.Dropout(p5)
        self.drop6 = nn.Dropout(p6)
        self.drop7 = nn.Dropout(p7)

    def encode(self, x):
        h1 = self.drop1(self.bn1(self.tanh(self.en1(x))))
        h2 = self.drop2(self.bn2(self.tanh(self.en2(h1))))
        h3 = self.drop3(self.bn3(self.tanh(self.en3(h2))))
        z = self.en4(h3)
        return z

    def decode(self, x):
        h5 = self.drop5(self.bn6(self.tanh(self.de1(self.drop4(self.bn5(self.tanh(x)))))))
        h6 = self.drop6(self.bn7(self.tanh(self.de2(h5))))
        h7 = self.drop7(self.bn8(self.tanh(self.de3(h6))))
        return self.de4(h7)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        pass


class AE_3D_200_bn_drop(nn.Module):
    def __init__(self, n_features=4, dropout=0):
        super(AE_3D_200_bn_drop, self).__init__()
        if type(dropout) is list:
            p1 = dropout[0]
            p2 = dropout[1]
            p3 = dropout[2]
            p4 = dropout[3]
            p5 = dropout[4]
            p6 = dropout[5]
            p7 = dropout[6]
        else:
            p1 = dropout
            p2 = dropout
            p3 = dropout
            p4 = dropout
            p5 = dropout
            p6 = dropout
            p7 = dropout
        self.en1 = nn.Linear(n_features, 200)
        self.bn1 = nn.BatchNorm1d(200)
        self.en2 = nn.Linear(200, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.en3 = nn.Linear(100, 50)
        self.bn3 = nn.BatchNorm1d(50)
        self.en4 = nn.Linear(50, 3)
        self.bn5 = nn.BatchNorm1d(3)
        self.de1 = nn.Linear(3, 50)
        self.bn6 = nn.BatchNorm1d(50)
        self.de2 = nn.Linear(50, 100)
        self.bn7 = nn.BatchNorm1d(100)
        self.de3 = nn.Linear(100, 200)
        self.bn8 = nn.BatchNorm1d(200)
        self.de4 = nn.Linear(200, n_features)
        self.tanh = nn.Tanh()
        self.drop1 = nn.Dropout(p1)
        self.drop2 = nn.Dropout(p2)
        self.drop3 = nn.Dropout(p3)
        self.drop4 = nn.Dropout(p4)
        self.drop5 = nn.Dropout(p5)
        self.drop6 = nn.Dropout(p6)
        self.drop7 = nn.Dropout(p7)

    def encode(self, x):
        h1 = self.drop1(self.bn1(self.tanh(self.en1(x))))
        h2 = self.drop2(self.bn2(self.tanh(self.en2(h1))))
        h3 = self.drop3(self.bn3(self.tanh(self.en3(h2))))
        z = self.en4(h3)
        return z

    def decode(self, x):
        h5 = self.drop5(self.bn6(self.tanh(self.de1(self.drop4(self.bn5(self.tanh(x)))))))
        h6 = self.drop6(self.bn7(self.tanh(self.de2(h5))))
        h7 = self.drop7(self.bn8(self.tanh(self.de3(h6))))
        return self.de4(h7)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        pass


class AE_3D_500cone_bn(nn.Module):
    def __init__(self, n_features=4):
        super(AE_3D_500cone_bn, self).__init__()
        self.en1 = nn.Linear(n_features, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.en2 = nn.Linear(400, 200)
        self.bn2 = nn.BatchNorm1d(200)
        self.en3 = nn.Linear(200, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.en4 = nn.Linear(100, 3)
        self.bn5 = nn.BatchNorm1d(3)
        self.de1 = nn.Linear(3, 100)
        self.bn6 = nn.BatchNorm1d(100)
        self.de2 = nn.Linear(100, 100)
        self.bn7 = nn.BatchNorm1d(100)
        self.de3 = nn.Linear(100, 100)
        self.bn8 = nn.BatchNorm1d(100)
        self.de4 = nn.Linear(100, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        h1 = self.bn1(self.tanh(self.en1(x)))
        h2 = self.bn2(self.tanh(self.en2(h1)))
        h3 = self.bn3(self.tanh(self.en3(h2)))
        z = self.en4(h3)
        return z

    def decode(self, x):
        h5 = self.bn6(self.tanh(self.de1(self.bn5(self.tanh(x)))))
        h6 = self.bn7(self.tanh(self.de2(h5)))
        h7 = self.bn8(self.tanh(self.de3(h6)))
        return self.de4(h7)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        pass


class AE_big_2D_v1(nn.Module):
    def __init__(self, n_features=4):
        super(AE_big_2D_v1, self).__init__()
        self.en1 = nn.Linear(n_features, 8)
        self.en2 = nn.Linear(8, 6)
        self.en3 = nn.Linear(6, 4)
        self.en4 = nn.Linear(4, 2)
        self.de1 = nn.Linear(2, 4)
        self.de2 = nn.Linear(4, 6)
        self.de3 = nn.Linear(6, 8)
        self.de4 = nn.Linear(8, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-8-6-4-2-4-6-8-out'


class AE_big_2D_v2(nn.Module):
    def __init__(self, n_features=4):
        super(AE_big_2D_v2, self).__init__()
        self.en1 = nn.Linear(n_features, 8)
        self.en2 = nn.Linear(8, 6)
        self.en3 = nn.Linear(6, 4)
        self.en4 = nn.Linear(4, 3)
        self.en5 = nn.Linear(3, 2)
        self.de1 = nn.Linear(2, 3)
        self.de2 = nn.Linear(3, 4)
        self.de3 = nn.Linear(4, 6)
        self.de4 = nn.Linear(6, 8)
        self.de5 = nn.Linear(8, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en5(self.tanh(self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))))

    def decode(self, x):
        return self.de5(self.tanh(self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-8-6-4-3-2-3-4-6-8-out'


class AE_2D(nn.Module):
    def __init__(self, n_features=4):
        super(AE_2D, self).__init__()
        self.en1 = nn.Linear(n_features, 20)
        self.en2 = nn.Linear(20, 10)
        self.en3 = nn.Linear(10, 6)
        self.en4 = nn.Linear(6, 2)
        self.de1 = nn.Linear(2, 6)
        self.de2 = nn.Linear(6, 10)
        self.de3 = nn.Linear(10, 20)
        self.de4 = nn.Linear(20, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-20-10-6-2-6-10-20-out'


class AE_2D_v2(nn.Module):
    def __init__(self, n_features=4):
        super(AE_2D_v2, self).__init__()
        self.en1 = nn.Linear(n_features, 50)
        self.en2 = nn.Linear(50, 20)
        self.en3 = nn.Linear(20, 10)
        self.en4 = nn.Linear(10, 2)
        self.de1 = nn.Linear(2, 10)
        self.de2 = nn.Linear(10, 20)
        self.de3 = nn.Linear(20, 50)
        self.de4 = nn.Linear(50, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-50-20-10-2-10-20-50-out'


class AE_big_2D_v3(nn.Module):
    def __init__(self, n_features=4):
        super(AE_big_2D_v3, self).__init__()
        self.en1 = nn.Linear(n_features, 8)
        self.en2 = nn.Linear(8, 6)
        self.en3 = nn.Linear(6, 2)
        self.de1 = nn.Linear(2, 6)
        self.de2 = nn.Linear(6, 8)
        self.de3 = nn.Linear(8, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))

    def decode(self, x):
        return self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-8-6-2-6-8-out'


class AE_2D_v3(nn.Module):
    def __init__(self, n_features=4):
        super(AE_2D_v3, self).__init__()
        self.en1 = nn.Linear(n_features, 100)
        self.en2 = nn.Linear(100, 200)
        self.en3 = nn.Linear(200, 100)
        self.en4 = nn.Linear(100, 2)
        self.de1 = nn.Linear(2, 100)
        self.de2 = nn.Linear(100, 200)
        self.de3 = nn.Linear(200, 100)
        self.de4 = nn.Linear(100, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-100-200-100-2-100-200-100-out'


class AE_2D_v4(nn.Module):
    def __init__(self, n_features=4):
        super(AE_2D_v4, self).__init__()
        self.en1 = nn.Linear(n_features, 500)
        self.en2 = nn.Linear(500, 200)
        self.en3 = nn.Linear(200, 100)
        self.en4 = nn.Linear(100, 2)
        self.de1 = nn.Linear(2, 100)
        self.de2 = nn.Linear(100, 200)
        self.de3 = nn.Linear(200, 500)
        self.de4 = nn.Linear(500, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-500-200-100-2-100-200-500-out'


class AE_2D_v5(nn.Module):
    def __init__(self, n_features=4):
        super(AE_2D_v5, self).__init__()
        self.en1 = nn.Linear(n_features, 200)
        self.en2 = nn.Linear(200, 100)
        self.en3 = nn.Linear(100, 50)
        self.en4 = nn.Linear(50, 2)
        self.de1 = nn.Linear(2, 50)
        self.de2 = nn.Linear(50, 100)
        self.de3 = nn.Linear(100, 200)
        self.de4 = nn.Linear(200, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-200-100-50-2-50-100-200-out'


class AE_2D_v100(nn.Module):
    def __init__(self, n_features=4):
        super(AE_2D_v100, self).__init__()
        self.en1 = nn.Linear(n_features, 100)
        self.en2 = nn.Linear(100, 100)
        self.en3 = nn.Linear(100, 100)
        self.en4 = nn.Linear(100, 2)
        self.de1 = nn.Linear(2, 100)
        self.de2 = nn.Linear(100, 100)
        self.de3 = nn.Linear(100, 100)
        self.de4 = nn.Linear(100, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-100-100-100-2-100-100-100-out'


class AE_2D_v50(nn.Module):
    def __init__(self, n_features=4):
        super(AE_2D_v50, self).__init__()
        self.en1 = nn.Linear(n_features, 50)
        self.en2 = nn.Linear(50, 50)
        self.en3 = nn.Linear(50, 50)
        self.en4 = nn.Linear(50, 2)
        self.de1 = nn.Linear(2, 50)
        self.de2 = nn.Linear(50, 50)
        self.de3 = nn.Linear(50, 50)
        self.de4 = nn.Linear(50, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-50-50-50-2-50-50-50-out'


class AE_2D_v1000(nn.Module):
    def __init__(self, n_features=4):
        super(AE_2D_v1000, self).__init__()
        self.en1 = nn.Linear(n_features, 1000)
        self.en2 = nn.Linear(1000, 400)
        self.en3 = nn.Linear(400, 100)
        self.en4 = nn.Linear(100, 2)
        self.de1 = nn.Linear(2, 100)
        self.de2 = nn.Linear(100, 400)
        self.de3 = nn.Linear(400, 1000)
        self.de4 = nn.Linear(1000, n_features)
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-1000-400-100-2-100-400-1000-out'


# Some helper functions
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, pin_memory=True),
        DataLoader(valid_ds, batch_size=bs * 2, pin_memory=True),
    )


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl, device):
    since = time.time()
    epochs_train_loss = []
    epochs_val_loss = []
    for epoch in range(epochs):
        running_train_loss = 0.
        epoch_start = time.perf_counter()
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            loss, lenxb = loss_batch(model, loss_func, xb, yb, opt)
            running_train_loss += np.multiply(loss, lenxb)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb_tmp.to(device), yb_tmp.to(device)) for xb_tmp, yb_tmp in valid_dl]
            )
        train_loss = running_train_loss / len(train_dl.dataset)
        epochs_train_loss.append(train_loss)
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        epochs_val_loss.append(val_loss)
        if(epoch % 1 == 0):
            current_time = time.perf_counter()
            delta_t = current_time - epoch_start
            # print('Epoch ' + str(epoch) + ':', 'Validation loss = ' + str(val_loss) + ' Time: ' + str(datetime.timedelta(seconds=delta_t)))
            print('Epoch: {:d} Train Loss: {:.3e} Val Loss: {:.3e}, Time: {}'.format(epoch, train_loss, val_loss, str(datetime.timedelta(seconds=delta_t))))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return pd.DataFrame({'Epoch': np.arange(epochs), 'train_loss': np.array(epochs_train_loss), 'val_loss': np.array(epochs_val_loss), 'epoch_time': delta_t})


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss

