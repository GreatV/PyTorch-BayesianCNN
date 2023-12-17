from __future__ import print_function


import torch

import config_bayesian as cfg
from models.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from models.BayesianModels.BayesianAlexNet import BBBAlexNet
from models.BayesianModels.BayesianLeNet import BBBLeNet

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getModel(net_type, inputs, outputs, priors, layer_type, activation_type):
    if net_type == "lenet":
        return BBBLeNet(outputs, inputs, priors, layer_type, activation_type)
    elif net_type == "alexnet":
        return BBBAlexNet(outputs, inputs, priors, layer_type, activation_type)
    elif net_type == "3conv3fc":
        return BBB3Conv3FC(outputs, inputs, priors, layer_type, activation_type)
    else:
        raise ValueError("Network should be either [LeNet / AlexNet / 3Conv3FC")


if __name__ == "__main__":
    # Hyper Parameter settings
    layer_type = cfg.layer_type
    activation_type = cfg.activation_type
    priors = cfg.priors

    inputs = 1
    outputs = 10
    net = getModel("lenet", inputs, outputs, priors, layer_type, activation_type)

    x = torch.randn(256, 1, 32, 32).to(device)
    net = net.to(device)
    net.eval()
    try:
        torch.export.export(net, (x,))
        print("[JIT] torch.export successed.")
        exit(0)
    except Exception as e:
        print("[JIT] rch.export failed.")
        raise e
