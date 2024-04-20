import numpy as np
import time

from keras import Input
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Model
from keras import backend as K

from DrAIve import DrAIve
from ReinforcementLearning import AgentTrainer


class PPOAgent:
    EPSILON = 0.2  # Clips loss between 1-e and 1+e
    EPOCHS = 10  # Number of times gradient Ascend is run on the updates
    ENTROPY_SCALAR = 0.001
    GAMMA = 0.99
    LEARNING_RATE = 0.001

    def __init__(self, model_name, input_shape, output_count, training_steps):
        self.model_name = model_name
        self.input_shape = input_shape
        self.output_count = output_count
        self.training_steps = training_steps

        self.dummy_actions = np.zeros((1, output_count))
        self.dummy_value = np.zeros((1, 1))

        self.actor = self.create_actor(input_shape, output_count)
        self.critic = self.create_critic(input_shape)

    def create_actor(self, input_shape, output_count):
        input = Input(shape=input_shape)
        advantage = Input((1,))
        old_prediction = Input((output_count,))
        actor_dense = Dense(64, activation='relu')(input)
        actor_dense2 = Dense(128, activation='relu')(actor_dense)
        actor_dense3 = Dense(64, activation='relu')(actor_dense2)
        output = Dense(output_count, activation="softmax")(actor_dense3)

        actor = Model([input, advantage, old_prediction], output)
        actor.compile(optimizer=Adam(lr=self.LEARNING_RATE), loss=[self.ppo_loss(advantage=advantage,
                                                                                 old_prediction=old_prediction)])
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
        policy = self.actor.predict([np.array(state).reshape(-1, *self.input_shape), self.dummy_value, self.dummy_actions]).ravel()
        return np.random.choice(np.arange(self.output_count), 1, p=policy)[0], policy

    def train(self, states, actions, policy, rewards, next_states, dones):
        targets = np.zeros((len(states), 1))
        advantages = np.zeros((len(states), 1))
        predictions = np.zeros((len(states), 1))

        for i in range(len(states)):
            reshaped_state = np.array(states[i]).reshape(-1, *self.input_shape)
            reshaped_next_state = np.array(next_states[i]).reshape(-1, *self.input_shape)
            value = self.critic.predict(reshaped_state)
            next_value = self.critic.predict(reshaped_next_state)

            predictions[i] = value

            if dones[i]:
                advantages[i] = rewards[i] - value
                targets[i][0] = rewards[i]
            else:
                advantages[i] = rewards[i] + self.GAMMA * next_value - value
                targets[i][0] = rewards[i] + self.GAMMA * next_value

        self.actor.fit([np.array(states), advantages, np.array(policy)], np.array(actions), verbose=0, epochs=self.EPOCHS)
        self.critic.fit(np.array(states), targets, verbose=0)

    def ppo_loss(self, advantage, old_prediction):
        def loss(y_true, y_pred):
            probability = K.sum(y_true * y_pred, axis=-1)
            old_probability = K.sum(y_true * old_prediction, axis=-1)
            ratio = probability/(old_probability + 1e-10)
            return -K.mean(K.minimum(ratio * advantage, K.clip(ratio, 1 - self.EPSILON, 1 + self.EPSILON) * advantage) +
                           self.ENTROPY_SCALAR * -(probability * K.log(probability + 1e-10)))
        return loss

    def save(self, average_reward, min_reward, max_reward):
        self.actor.save(f'models/{self.model_name}_Actor__{max_reward:>7.2f}max_'
                        f'{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
        self.critic.save(f'models/{self.model_name}_Critic__{max_reward:>7.2f}max_'
                         f'{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')


class PPOAgentTrainer(AgentTrainer):
    def __init__(self, env, model_name, training_steps):
        AgentTrainer.__init__(self, env)
        self.agent = PPOAgent(model_name, env.OUTPUT_SHAPE, env.ALLOWED_INPUTS, training_steps)

    def run(self, episodes):
        game_number = 1
        game_rewards = []
        aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

        toggle_show = False
        toggle_limit_fps = False

        states, actions, policies, rewards, next_states, dones = [], [], [], [], [], []

        while game_number <= episodes:
            done = False
            game_reward = 0
            current_state = self.env.reset()

            while not done:  # Game loop
                if self.show_track_forced:
                    self.env.render()
                toggle_show, toggle_limit_fps = self.handle_keyboard(toggle_show, toggle_limit_fps)
                action, policy = self.agent.get_action(current_state)
                new_states, action_rewards, done = self.env.step([action])

                states.append(current_state)
                actions.append(action)
                policies.append(policy)
                rewards.append(action_rewards[0])
                next_states.append(new_states[0])
                dones.append(done)

                if not len(states) % self.agent.training_steps:
                    self.agent.train(states, actions, policies, rewards, next_states, dones)
                    states, actions, policies, rewards, next_states, dones = [], [], [], [], [], []

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


# PPOAgentTrainer(DrAIve(1), "DrAIve-PPO-25Step", 25).run(2500)
PPOAgentTrainer(DrAIve(1), "DrAIve-PPO-50Step", 50).run(2500)
PPOAgentTrainer(DrAIve(1), "DrAIve-PPO-100Step", 100).run(2500)
PPOAgentTrainer(DrAIve(1), "DrAIve-PPO-250Step", 250).run(2500)
PPOAgentTrainer(DrAIve(1), "DrAIve-PPO-500Step", 500).run(2500)
PPOAgentTrainer(DrAIve(1), "DrAIve-PPO-1000Step", 1000).run(2500)
