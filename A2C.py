import numpy as np
import time
import keras.backend as K

from keras import Input
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Dense
from keras.models import Model

from DrAIve import DrAIve
from ReinforcementLearning import AgentTrainer


class Agent:
    def __init__(self, input_shape, output_count, lr):
        self.input_shape = input_shape
        self.output_count = output_count
        self.optimizer = Adam(lr=lr)

    def fit(self, state, target):
        self.model.fit(self.reshape(state), target, epochs=1, verbose=0)

    def predict(self, state):
        return self.model.predict(self.reshape(state))

    def reshape(self, x):
        return np.array(x).reshape(-1, *self.input_shape)


class Actor(Agent):
    def __init__(self, input_shape, output_count, lr):
        Agent.__init__(self, input_shape, output_count, lr)

        self.model = self.create_model(input_shape, output_count)
        self.train = self.create_train_function()

    def create_model(self, input_shape, output_count):
        input = Input(shape=input_shape)
        actor_dense = Dense(64, activation='relu')(input)
        actor_dense2 = Dense(128, activation='relu')(actor_dense)
        actor_dense3 = Dense(64, activation='relu')(actor_dense2)
        output = Dense(output_count, activation="softmax")(actor_dense3)
        return Model(input, output)

    def create_train_function(self):
        # Create optimizer with entropy factor to encourage exploration
        states = self.model.input
        actions_placeholder = K.placeholder(shape=(None, self.output_count))
        advantages_placeholder = K.placeholder(shape=(None,))

        weighted_actions = K.sum(actions_placeholder * self.model.output, axis=1)
        action_losses = K.log(weighted_actions + 1e-10) * K.stop_gradient(advantages_placeholder)
        entropy = K.sum(self.model.output * K.log(self.model.output + 1e-10), axis=1)

        loss = 0.001 * entropy - K.sum(action_losses)

        updates = self.optimizer.get_updates(self.model.trainable_weights, [], loss)

        return K.function([states, actions_placeholder, advantages_placeholder], [], updates=updates)


class Critic(Agent):
    def __init__(self, input_shape, output_count, lr):
        Agent.__init__(self, input_shape, output_count, lr)

        self.model = self.create_model(input_shape)
        self.train = self.create_train_function()

    def create_model(self, input_shape):
        input = Input(shape=input_shape)
        actor_dense = Dense(64, activation='relu')(input)
        actor_dense2 = Dense(128, activation='relu')(actor_dense)
        actor_dense3 = Dense(64, activation='relu')(actor_dense2)
        output = Dense(1, activation="linear")(actor_dense3)
        return Model(input, output)

    def create_train_function(self):
        states = self.model.input
        discounted_reward = K.placeholder(shape=(None,))
        critic_loss = K.mean(K.square(discounted_reward - self.model.output))
        updates = self.optimizer.get_updates(self.model.trainable_weights, [], critic_loss)
        return K.function([states, discounted_reward], [], updates=updates)


class A2CAgent:
    GAMMA = 0.99
    LEARNING_RATE = 0.001

    def __init__(self, model_name, input_shape, output_count):
        self.model_name = model_name
        self.input_shape = input_shape
        self.output_count = output_count

        self.actor = Actor(input_shape, output_count, self.LEARNING_RATE)
        self.critic = Critic(input_shape, output_count, self.LEARNING_RATE)

    def get_action(self, state):
        return np.random.choice(np.arange(self.output_count), 1, p=self.actor.predict(state).ravel())[0]

    def calc_discounted_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        cumulative_reward = 0

        for t in reversed(range(0, len(rewards))):
            cumulative_reward = rewards[t] + cumulative_reward * self.GAMMA
            discounted_rewards[t] = cumulative_reward

        return discounted_rewards

    def train_models(self, states, actions, rewards):
        discounted_rewards = self.calc_discounted_rewards(rewards)
        state_values = self.critic.predict(np.array(states))

        advantages = discounted_rewards - np.reshape(state_values, len(state_values))

        self.actor.train([states, actions, advantages])
        self.critic.train([states, discounted_rewards])

    def save(self, average_reward, min_reward, max_reward):
        self.actor.model.save(f'models/{self.model_name}_Actor__{max_reward:>7.2f}max_'
                              f'{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
        self.critic.model.save(f'models/{self.model_name}_Critic__{max_reward:>7.2f}max_'
                               f'{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')


class A2CAgentTrainer(AgentTrainer):
    def __init__(self, env, model_name):
        AgentTrainer.__init__(self, env)
        self.agent = A2CAgent(model_name, env.OUTPUT_SHAPE, env.ALLOWED_INPUTS)

    def run(self, episodes):
        game_number = 1
        game_rewards = []
        aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

        toggle_show = False
        toggle_limit_fps = False

        while game_number <= episodes:
            done = False
            game_reward = 0
            current_state = self.env.reset()

            actions, states, rewards = [], [], []

            while not done:  # Game loop
                if self.show_track_forced:
                    self.env.render()
                toggle_show, toggle_limit_fps = self.handle_keyboard(toggle_show, toggle_limit_fps)
                action = self.agent.get_action(current_state)
                new_states, action_rewards, done = self.env.step([action])

                actions.append(to_categorical(action, self.agent.output_count))
                rewards.append(action_rewards[0])
                states.append(current_state)

                game_reward += action_rewards[0]
                current_state = new_states[0]

            self.agent.train_models(states, actions, rewards)

            print("Game number: {}, reward: {}".format(game_number, game_reward))

            game_rewards.append(game_reward)

            if not game_number % self.AGGREGATE_STATS_EVERY or game_number == 1:
                average_reward, min_reward, max_reward = \
                    self.update_aggregate_rewards(game_number, game_rewards, aggr_ep_rewards)
                self.agent.save(average_reward, min_reward, max_reward)

            game_number += 1

        self.plot_rewards(aggr_ep_rewards, self.agent.model_name)
        self.env.stop()


A2CAgentTrainer(DrAIve(1), "DrAIve-A2C-Basic").run(2500)
