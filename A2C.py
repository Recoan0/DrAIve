import numpy as np
import time

from keras import Input
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Dense
from keras.models import Model

from DrAIve import DrAIve
from ReinforcementLearning import AgentTrainer


class A2CAgent:
    GAMMA = 0.99
    LEARNING_RATE = 0.001
    TRAIN_STEPS = 50

    def __init__(self, model_name, input_shape, output_count):
        self.model_name = model_name
        self.input_shape = input_shape
        self.output_count = output_count

        self.actor = self.create_actor(input_shape, output_count)
        self.critic = self.create_critic(input_shape)

    def create_actor(self, input_shape, output_count):
        input = Input(shape=input_shape)
        actor_dense = Dense(64, activation='relu')(input)
        actor_dense2 = Dense(128, activation='relu')(actor_dense)
        actor_dense3 = Dense(64, activation='relu')(actor_dense2)
        output = Dense(output_count, activation="softmax")(actor_dense3)

        actor = Model(input, output)
        actor.compile(optimizer=Adam(lr=self.LEARNING_RATE), loss="categorical_crossentropy")
        return actor

    def create_critic(self, input_shape):
        input = Input(shape=input_shape)
        actor_dense = Dense(64, activation='relu')(input)
        actor_dense2 = Dense(128, activation='relu')(actor_dense)
        actor_dense3 = Dense(64, activation='relu')(actor_dense2)
        output = Dense(1, activation="linear")(actor_dense3)

        critic = Model(input, output)
        critic.compile(optimizer=Adam(lr=self.LEARNING_RATE), loss="mse")
        return critic

    def get_action(self, state):
        policy = self.actor.predict(np.array(state).reshape(-1, *self.input_shape)).ravel()
        return np.random.choice(np.arange(self.output_count), 1, p=policy)[0]

    def train(self, states, actions, rewards, next_states, dones):
        targets = np.zeros((len(states), 1))
        advantages = np.zeros((len(states), self.output_count))
        for i in range(len(states)):
            reshaped_state = np.array(states[i]).reshape(-1, *self.input_shape)
            reshaped_next_state = np.array(next_states[i]).reshape(-1, *self.input_shape)
            value = self.critic.predict(reshaped_state)
            next_value = self.critic.predict(reshaped_next_state)

            if dones[i]:
                advantages[i][actions[i]] = rewards[i] - value
                targets[i][0] = rewards[i]
            else:
                advantages[i][actions[i]] = rewards[i] + self.GAMMA * next_value - value
                targets[i][0] = rewards[i] + self.GAMMA * next_value

        self.actor.fit(np.array(states), advantages, verbose=0)
        self.critic.fit(np.array(states), targets, verbose=0)

    def save(self, average_reward, min_reward, max_reward):
        self.actor.save(f'models/{self.model_name}_Actor__{max_reward:>7.2f}max_'
                        f'{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
        self.critic.save(f'models/{self.model_name}_Critic__{max_reward:>7.2f}max_'
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

        actions, states, rewards, next_states, dones = [], [], [], [], []

        while game_number <= episodes:
            done = False
            game_reward = 0
            current_state = self.env.reset()

            while not done:  # Game loop
                if self.show_track_forced:
                    self.env.render()
                toggle_show, toggle_limit_fps = self.handle_keyboard(toggle_show, toggle_limit_fps)
                action = self.agent.get_action(current_state)
                new_states, action_rewards, done = self.env.step([action])

                states.append(current_state)
                actions.append(action)
                rewards.append(action_rewards[0])
                next_states.append(new_states[0])
                dones.append(done)

                if not len(states) % self.agent.TRAIN_STEPS:
                    self.agent.train(states, actions, rewards, next_states, dones)
                    actions, states, rewards, next_states, dones = [], [], [], [], []

                game_reward += action_rewards[0]
                current_state = new_states[0]

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
