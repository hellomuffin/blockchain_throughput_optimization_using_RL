import numpy as np
from agent import ddpg
from env import MecBCEnv
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = MecBCEnv()
    agent = ddpg(alpha=0.0001, beta=0.001, input_dims=env.state.shape, gamma=0.99, tau=1.0, env=env, batch_size=64,
                 layer1_size=128, layer2_size=128, n_actions=3)

    score_history = []
    action_1_history = []
    action_2_history = []
    action_3_history = []
    punish_history = []
    best_score = 0
    np.random.seed(0)

    for i in range(100000):  # episode
        obs = env.reset()
        done = False
        score = 0
        reward = 0
        action = [2, 6, 1]
        average_action = [0, 0, 0]
        punish = 0
        while not done:
            act = agent.choose_action(obs, action)
            # act = [1, 1, -1]
            # print(act)
            new_state, reward, done, C2, action = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            if C2 == True:
                score += action[0] * action[1] / action[2]
            else:
                punish += 1
            obs = new_state
            average_action = average_action + action / 100

        action_1_history.append(average_action[0])
        action_2_history.append(average_action[1])
        action_3_history.append(average_action[2])

        score_history.append(score)
        punish_history.append(punish)

        # avg_score = np.mean(score_history[-100:])

        # if avg_score > best_score:
        #     best_score = avg_score
        #     # if not load_checkpoint:
        #     agent.save_models()

        print('episode', i, 'score  %.2f' % score, '100 game average %.2f' % np.mean(score_history[-100:]))

        plt.figure(1)
        plt.clf()
        plt.title('Training Process')
        plt.xlabel('Episode')
        plt.ylabel('Total reward')
        plt.plot(np.array(score_history))
        plt.savefig("rewards.png")

        plt.figure(2)
        plt.clf()
        plt.title('Action History')
        plt.xlabel('Episode')
        plt.ylabel('Actions')
        plt.plot(np.array(action_1_history))
        plt.plot(np.array(action_2_history))
        plt.plot(np.array(action_3_history))
        plt.savefig("actions.png")

        plt.figure(3)
        plt.clf()
        plt.title('Punish History')
        plt.xlabel('Episode')
        plt.ylabel('Punishes')
        plt.plot(np.array(punish_history))
        plt.savefig("punishes.png")

        # agent.save_models()

