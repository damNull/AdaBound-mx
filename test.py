import mxnet as mx
import torch
import numpy as np
from mxnet import autograd as ag_mx
from tqdm import tqdm

from adabound_mx import AdaBound as AdaBoundMX
from adabound_torch import AdaBound as AdaBoundT

import matplotlib.pyplot as plt


mx_net = mx.gluon.nn.Sequential()
with mx_net.name_scope():
    mx_net.add(mx.gluon.nn.Dense(1, in_units=12, use_bias=False))
    #mx_net.add(mx.gluon.nn.Dense(1, in_units=6, use_bias=False))

torch_net = torch.nn.Sequential(
    torch.nn.Linear(12, 1, bias=False),
    #torch.nn.Linear(6, 1, bias=False)
)

mx_net.initialize()
eps = 1e-6

if __name__ == '__main__':
    l1_param = np.random.normal(size=(1, 12))
    l2_param = np.random.normal(size=(1, 6))
    input_data = np.random.normal(size=(1, 12))

    # init mxnet network params
    mx_net[0].weight.data()[:] = mx.nd.array(l1_param)
    # mx_net[1].weight.data()[:] = mx.nd.array(l2_param)
    # init torch network params
    torch_net[0].weight.data[:] = torch.Tensor(l1_param)
    # torch_net[1].weight.data[:] = torch.Tensor(l2_param)
    # mx optimizer
    mx_opt = AdaBoundMX()
    mx_trainer = mx.gluon.Trainer(optimizer=mx_opt, params=mx_net.collect_params())
    # torch optimizer
    torch_opt = AdaBoundT(torch_net.parameters())
    # input data
    mx_input = mx.nd.array(input_data)
    torch_input = torch.Tensor(input_data)
    abs_diff = []
    for i in tqdm(range(100000)):
        # mx forward backward
        with ag_mx.record():
            mx_out = mx_net(mx_input)
        mx_out.backward()
        # torch forward backward
        torch_opt.zero_grad()
        torch_out = torch_net(torch_input)
        torch_out.backward()
        mx_out = mx_out[0].asscalar()
        torch_out = torch_out.item()
        # step
        torch_opt.step()
        mx_trainer.step(1)
        # check value
        #assert abs(mx_out - torch_out) < eps, 'not equal in iter %d' % i
        abs_diff.append(abs(mx_out - torch_out))
    plt.plot(abs_diff)
    plt.show()
    pass