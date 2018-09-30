"""Entry point to evolving the neural network. Start here."""
import gym
import numpy as np
from keras.models import Sequential
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from keras.optimizers import Adam
from rl.core import Processor
from keras.layers import Dense, Activation, Flatten
from game_interface import Game


DIRECTIONS = ['up', 'down', 'left', 'right']


class MyProcessor(Processor):
    def process_observation(self, observation):
        return np.array(observation)

    def process_state_batch(self, batch):
        return batch[0]

    def process_reward(self, reward):
        return reward


class Env2014(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.game = Game()

    def step(self, action):
        ob = self.game.elements
        reward = self.game.score
        self.game.move(DIRECTIONS[action])
        ob2 = self.game.elements
        return ob2, self.game.score - reward, ob == ob2, {}

    def reset(self):
        self.game = Game()
        return self.game.elements

    def render(self, mode='human'):
        pass


def main():
    env = Env2014()
    model = Sequential()
    model.add(Dense(22, input_shape=(16,), activation='sigmoid'))
    model.add(Dense(4))

    print(model.summary())
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=1000000)

    memory = SequentialMemory(window_length=1, limit=1000000)
    space = gym.spaces.Discrete(4)
    dqn = DQNAgent(model=model, nb_actions=space.n, policy=policy, memory=memory,
                   nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
                   train_interval=4, delta_clip=1., processor=MyProcessor())

    dqn.compile(Adam(lr=.00025), metrics=['mae'])
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format('123')
    checkpoint_weights_filename = 'dqn_' + '123' + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format('123')
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)

if __name__ == '__main__':
    main()
