"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""

import numpy as np
import torch
import torch.nn as nn
from utls import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
from torch import FloatTensor, LongTensor, ByteTensor
from env import MecBCEnv1
import os

os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 10000

# print(N_S, N_A)
Env = MecBCEnv1()
N_S = len(Env.input_state)
N_A = len(Env.Kspace) * len(Env.SMspace) * len(Env.DIspace)


# print(N_A)

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        # print(logits)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        # print(m)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)  # local network
        self.env = MecBCEnv1()
        self.Kcount = len(self.env.Kspace)
        self.SMcount = len(self.env.SMspace)
        self.DIcount = len(self.env.DIspace)

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.input_state
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                # if self.name == 'w00':
                #     self.env.render()
                a = self.lnet.choose_action(v_wrap(np.array(s)[None, :]))
                real_act = [self.env.SMspace[a // (self.Kcount * self.DIcount)],
                            self.env.Kspace[a % (self.Kcount * self.DIcount) // self.DIcount],
                            self.env.DIspace[a % self.DIcount]]
                s_, r, done, _ = self.env.step(real_act)
                # print(s_, r, done)
                if done: r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        print(ep_r)
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)

if __name__ == "__main__":
    gnet = Net(N_S, N_A)        # global network, 动作空间状态空间维度
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))      # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
            # Open a file with access mode 'a'
            file_object = open('a3c.txt', 'a')
            # Append 'hello' at the end of file
            file_object.write(str(r))
            file_object.write(" ")
            # Close the file
            file_object.close()

            import matplotlib.pyplot as plt
            plt.figure(1)
            plt.plot(res)
            plt.ylabel('Moving average ep reward')
            plt.xlabel('Step')
            plt.title('Training Process')
            plt.savefig("rewards.png")
            plt.show()


        else:
            break
    [w.join() for w in workers]


