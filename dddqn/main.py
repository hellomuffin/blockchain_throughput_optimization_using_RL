import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import gym
from imp import reload
import env

reload(env)


def train(DM, SI):
    # 超参数
    BATCH_SIZE = 32
    LR = 0.01  # learning rate
    EPSILON = 0.9  # 最优选择动作百分比
    GAMMA = 0.9  # 奖励递减参数
    TARGET_REPLACE_ITER = 100  # Q 现实网络的更新频率
    MEMORY_CAPACITY = 500  # 记忆库大小
    # env = gym.make('CartPole-v0')   # 立杆子游戏
    # env = env.unwrapped
    from env import MecBCEnv1
    env = MecBCEnv1(DM, SI)
    N_ACTIONS = len(env.Kspace) * len(env.SMspace) * len(env.DIspace)
    N_STATES = len(env.input_state)  # 杆子能获取的环境信息数
    d = len(env.DIspace)
    kk = len(env.Kspace)

    class Net(nn.Module):
        def __init__(self, ):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(N_STATES, 100)
            self.fc1.weight.data.normal_(0, 0.1)  # initialization
            self.out = nn.Linear(100, N_ACTIONS + 1)
            self.out.weight.data.normal_(0, 0.1)  # initialization

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            actions_value = self.out(x)
            return actions_value

    class DQN(object):
        def __init__(self):
            self.eval_net, self.target_net = Net(), Net()

            self.learn_step_counter = 0  # 用于 target 更新计时
            self.memory_counter = 0  # 记忆库记数
            self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # 初始化记忆库
            self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # torch 的优化器
            self.loss_func = nn.MSELoss()  # 误差公式

        def choose_action(self, x):
            x = torch.unsqueeze(torch.FloatTensor(x), 0)
            # 这里只输入一个 sample
            if np.random.uniform() < EPSILON:  # 选最优动作
                actions_value = self.eval_net.forward(x)
                # print(actions_value)
                # print(torch.max(actions_value, 1)[1].data)
                action = torch.max(actions_value[:, :-1], 1)[1].data.numpy()[0]  # return the argmax
            else:  # 选随机动作
                action = np.random.randint(0, N_ACTIONS)
            return action

        def store_transition(self, s, a, r, s_):
            transition = np.hstack((s, [a, r], s_))
            # 如果记忆库满了, 就覆盖老数据
            index = self.memory_counter % MEMORY_CAPACITY
            self.memory[index, :] = transition
            self.memory_counter += 1

        def learn(self):
            # target net 参数更新
            if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
                self.target_net.load_state_dict(self.eval_net.state_dict())
            self.learn_step_counter += 1

            # 抽取记忆库中的批数据
            sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
            b_memory = self.memory[sample_index, :]
            b_s = torch.FloatTensor(b_memory[:, :N_STATES])
            b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
            b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
            b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

            # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
            duelling_net_eval = self.eval_net(b_s)
            state_value_eval = duelling_net_eval[:, -1]
            avg_advantage_eval = torch.mean(duelling_net_eval[:, :-1], dim=1)
            q_values_eval = state_value_eval.unsqueeze(1) + (
                        duelling_net_eval[:, :-1] - avg_advantage_eval.unsqueeze(1))
            q_eval = q_values_eval.gather(1, b_a)
            # q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)

            max_action_indexes = self.eval_net(b_s_)[:, :-1].detach().argmax(1)
            duelling_net_output = self.target_net(b_s_)
            state_value = duelling_net_output[:, -1]
            avg_advantage = torch.mean(duelling_net_output[:, :-1], dim=1)
            q_values = state_value.unsqueeze(1) + (duelling_net_output[:, :-1] - avg_advantage.unsqueeze(1))
            q_next = q_values.gather(1, max_action_indexes.unsqueeze(1))

            # q_next = self.target_net(b_s_).gather(1, max_action_indexes.unsqueeze(1))

            # q_next = self.target_net(b_s_).detach()     # q_next 不进行反向传递误差, 所以 detach
            q_target = b_r + GAMMA * q_next.max(1)[0]  # shape (batch, 1)
            loss = self.loss_func(q_eval, q_target)

            step_loss.append(float(loss))

            # 计算, 更新 eval net
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    import matplotlib.pyplot as plt
    dqn = DQN()  # 定义 DQN 系统
    rewards = []
    step_loss = []
    result = 0

    moving_sum = 0
    max_average = 0;
    for i_episode in range(3000):
        env.reset()
        s = env.input_state
        re = 0
        while True:
            # env.render()    # 显示实验动画
            a = dqn.choose_action(s)
            real_act = [env.SMspace[a // (kk * d)], env.Kspace[a % (kk * d) // d], env.DIspace[a % d]]
            # print(real_act)
            # 选动作, 得到环境反馈
            s_, r, done, info = env.step(real_act)
            re += r

            # 修改 reward, 使 DQN 快速学习
            # x, x_dot, theta, theta_dot = s_
            #         r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            #         r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            #         r = r1 + r2

            # 存记忆
            dqn.store_transition(s, a, r, s_)

            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()  # 记忆库满了就进行学习

            if done:  # 如果回合结束, 进入下回合
                rewards.append(re)
                if len(rewards) >= 50:
                    resultt = np.mean(rewards[-50:])
                    if resultt > result:
                        result = resultt

                file_object = open('dddqn.txt', 'a')
                # Append 'hello' at the end of file
                file_object.write(str(re))
                file_object.write(" ")
                # Close the file
                file_object.close()



                # plt.figure(figsize = (10,5))
                plt.figure(1)
                plt.clf()
                plt.title('Training Process')
                plt.xlabel('Episode')
                plt.ylabel('Total reward')
                plt.title('Training process')
                plt.plot(rewards)
                plt.savefig("rewards.png")
                # plt.show()

                if i_episode != 0 and i_episode % 50 == 0:
                    print("maximum moving average ", end = "")
                    if moving_sum/50 > max_average:
                        max_average = moving_sum/50
                    print(max_average)
                    moving_sum = 0
                else:
                    moving_sum += rewards[len(rewards)-1]










                # plt.figure(figsize = (10,5))
                plt.figure(2)
                plt.clf()
                plt.title('Loss Function')
                plt.xlabel('Training step')
                plt.ylabel('Loss')
                plt.title('Loss during training process')
                plt.plot(step_loss)
                plt.savefig("loss.png")
                # plt.show()
                break

            s = s_
    print(result)
    res_avg = 0
    for i in range(0, len(result) - 50):
        temp = np.mean(result[i:i + 50])
        if (temp > res_avg): res_avg = temp
    print(res_avg)
    # env.close()


if __name__ == '__main__':
    train(None,None)
