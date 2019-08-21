import pygame
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import random
import os

from keras import backend as K, Input
from keras.models import load_model, Model
from keras.layers import Dense, Lambda, Subtract, Add
from keras.optimizers import Adam

from DrAIve import DrAIve


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes (final nodes) that contains experiences

        # Generate the tree with all nodes values = 0
        self.tree = np.zeros(2 * capacity - 1)
        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    beta = 0.4  # importance-sampling, from initial value increasing to 1

    beta_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree
        self.tree = SumTree(capacity)

    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected -> use minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)  # set the max p for new p

    def sample(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = []

        tree_idx, is_weights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n  # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.beta)

        for i in range(n):
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            index, priority, data = self.tree.get_leaf(value)

            # P(j)
            sampling_probabilities = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            is_weights[i, 0] = np.power(n * sampling_probabilities, -self.beta) / max_weight

            tree_idx[i] = index

            experience = data

            memory_b.append(experience)

        return tree_idx, memory_b, is_weights

    def batch_update(self, tree_idx, abs_errors):
        for i in range(len(abs_errors)):
            abs_errors[i] += self.e  # convert to abs and avoid 03
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class DQNAgent:
    REPLAY_MEMORY_SIZE = 50000  # How many last steps to keep for model training
    MIN_REPLAY_MEMORY_SIZE = 10000  # Minimum number of steps in memory to start training
    MINIBATCH_STANDARD_SIZE = 32  # How many steps/samples to use for training
    DISCOUNT = 0.99
    UPDATE_TARGET_EVERY = 5000  # Amount of steps
    MEMORY_FRACTION = 0.20
    LEARNING_RATE = 0.001

    # Exploration settings, current settings make epsilon reset about every 1000 runs
    EPSILON_DECAY = 0.9978603
    MIN_EPSILON = 0.01

    def __init__(self, model_name, double, dueling, input_shape, output_options, fit_every_steps):
        self.MODEL_NAME = model_name
        self.double = double
        self.dueling = dueling
        self.input_shape = input_shape
        self.output_options = output_options

        # main model, gets trained every step
        self.model = self.create_dueling_model() if self.dueling else self.create_standard_model()

        # Target model, this is what we .predict against every step
        self.target_model = self.create_dueling_model() if self.dueling else self.create_standard_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = Memory(self.REPLAY_MEMORY_SIZE)
        self.fit_every_steps = fit_every_steps
        self.minibatch_size = self.MINIBATCH_STANDARD_SIZE * fit_every_steps

        self.epsilon = 1

    def create_dueling_model(self):
        inputs = Input(shape=self.input_shape)
        hidden = Dense(64, input_shape=self.input_shape, activation="relu",
                       kernel_initializer=tf.variance_scaling_initializer(scale=2))(inputs)
        hidden2 = Dense(32, input_shape=self.input_shape, activation="relu",
                        kernel_initializer=tf.variance_scaling_initializer(scale=2))(hidden)
        state_value = Dense(1, activation="linear")(hidden2)
        advantage_values = Dense(self.output_options, activation="linear")(hidden2)
        mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(advantage_values)
        normalized_advantage_values = Subtract()([advantage_values, mean])
        outputs = Add()([state_value, normalized_advantage_values])
        print("Shape of the output layer: {}".format(outputs.shape))
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=tf.losses.huber_loss, optimizer=Adam(lr=self.LEARNING_RATE), metrics=['accuracy'])

        return model

    def create_standard_model(self):
        inputs = Input(shape=self.input_shape)
        hidden = Dense(64, input_shape=self.input_shape, activation="relu",
                       kernel_initializer=tf.variance_scaling_initializer(scale=2))(inputs)
        hidden2 = Dense(32, activation="relu", kernel_initializer=tf.variance_scaling_initializer(scale=2))(hidden)
        outputs = Dense(self.output_options, activation="linear")(hidden2)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=tf.losses.huber_loss, optimizer=Adam(lr=self.LEARNING_RATE), metrics=['accuracy'])

        return model

    def load_model(self, model):
        self.model = model
        self.target_model = model

    def update_replay_memory(self, transition):
        self.replay_memory.store(transition)

    def get_action(self, state):
        if np.random.random() > self.epsilon:
            return np.argmax(self.get_qs(state))
        else:
            return np.random.randint(0, self.output_options)

    def get_qs(self, state):
        # Reshape array into vector with x inputs (currently 12)
        return self.model.predict(np.array(state).reshape(-1, *self.input_shape))[0]

    def decay_epsilon(self):
        if self.epsilon > self.MIN_EPSILON:
            self.epsilon = max(self.MIN_EPSILON, self.epsilon * self.EPSILON_DECAY)
            return True
        return False

    def train(self, step):
        # if step < self.MIN_REPLAY_MEMORY_SIZE:  # Only if memory is Dequeue
        #     return step + 1

        tree_idx, minibatch, is_weights = self.replay_memory.sample(self.minibatch_size)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        if self.double:
            next_move_list = self.model.predict(new_current_states)  # FOR DOUBLE DQN

        X = []
        y = []
        absolute_errors = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                if not self.double:
                    future_q = np.max(future_qs_list[index])  # FOR REGULAR TARGETED DQN
                else:
                    # FOR DOUBLE DQN
                    model_selected_action = np.argmax(next_move_list[index])  # action selected by online model
                    future_q = future_qs_list[index][model_selected_action]  # action evaluated by target model

                new_q = reward + self.DISCOUNT * future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]

            absolute_errors.append(abs(current_qs[action] - new_q))
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.replay_memory.batch_update(tree_idx, absolute_errors)
        self.model.fit(np.array(X), np.array(y), batch_size=self.minibatch_size, verbose=0, shuffle=False)

        if step >= self.UPDATE_TARGET_EVERY:
            print("Updating target model!")
            self.target_model.set_weights(self.model.get_weights())
            return 0
        return step + 1


class QTrainer:
    TRACK_AMOUNT = 5
    TICKS = 30  # Number of steps per second the AI is asked to give an action

    DISPLAY_WIDTH = 1920
    DISPLAY_HEIGHT = 1080
    SHOW_EVERY = 50
    SHOW_FPS_EVERY = 1

    CONTINUE_AFTER_EPSILON_TARGET = 450
    AGGREGATE_STATS_EVERY = 50  # episodes
    FIT_EVERY_STEPS = 4

    STANDARD_TRACKS = True

    def __init__(self, env, model_name, double, dueling):
        pygame.init()
        self.screen = env.screen
        self.screen.set_alpha(None)
        self.clock = pygame.time.Clock()
        self.env = env
        self.agent = DQNAgent(model_name, double, dueling, env.OUTPUT_SHAPE, env.ALLOWED_INPUTS, self.FIT_EVERY_STEPS)
        self.show_track_forced = True
        self.limit_fps = False

    def run(self, load_model_name, epsilon, episodes):
        if load_model_name is not None:
            self.agent.load_model(load_model(load_model_name))
            self.agent.epsilon = epsilon
        game_number = 1
        game_rewards = []
        reset_epsilon_game = -1
        aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}
        pygame.display.set_caption('DrAIve')

        toggle_show = False
        toggle_limit_fps = False
        fit_network_counter = 0
        update_target_counter = 0

        dt = 1 / self.TICKS  # Standard delta time for consistent AI movement

        self.fill_agent_memory(dt, toggle_show, toggle_limit_fps)

        while game_number <= episodes:
            self.env.init()
            done = False
            game_reward = 0
            show_time = 0
            current_state = self.env.build_state(self.env.cars[0])

            while not done:  # Game loop
                pygame.event.get()  # Prevents OS from seeing game as not responding
                toggle_show, toggle_limit_fps = self.handle_keyboard(toggle_show, toggle_limit_fps)

                draw_screen, show_time = self.draw_screen(game_number, show_time)
                action = self.agent.get_action(current_state)
                new_states, rewards, done = self.env.step([action], dt, draw_screen)

                game_reward += rewards[0]
                if np.shape(current_state) != self.env.INPUT_SHAPE:
                    print("UNMATCHING CURRENT STATE FOUND")
                    print(current_state)
                elif np.shape(new_states[0]) != self.env.OUTPUT_SHAPE:
                    print("UNMATCHING NEW STATE FOUND")
                    print(current_state)
                self.agent.update_replay_memory((current_state, action, rewards[0], new_states[0], done))

                if fit_network_counter % self.FIT_EVERY_STEPS == 0:
                    fit_network_counter = 0
                    update_target_counter = self.agent.train(update_target_counter)
                fit_network_counter += 1
                current_state = new_states[0]

                if draw_screen:
                    if self.limit_fps:
                        self.clock.tick(self.TICKS)
                    else:
                        self.clock.tick()

            print("Game number: {}, reward: {}, epsilon: {}".format(game_number, game_reward, self.agent.epsilon))

            # if reward == self.FINISH_REWARD:
            #     print("Finished! Starting next track.")
            #     self.agent.epsilon = 1  # Reset epsilon to encourage learning the new track
            #     current_track = random.choice(self.tracks)  # Start on new track

            # Append game reward to a list and log stats (every given number of games)
            game_rewards.append(game_reward)
            if not game_number % self.AGGREGATE_STATS_EVERY or game_number == 1:
                average_reward = sum(game_rewards[-self.AGGREGATE_STATS_EVERY:]) \
                                 / len(game_rewards[-self.AGGREGATE_STATS_EVERY:])
                min_reward = min(game_rewards[-self.AGGREGATE_STATS_EVERY:])
                max_reward = max(game_rewards[-self.AGGREGATE_STATS_EVERY:])
                aggr_ep_rewards['ep'].append(game_number)
                aggr_ep_rewards['avg'].append(average_reward)
                aggr_ep_rewards['min'].append(min_reward)
                aggr_ep_rewards['max'].append(max_reward)
                self.agent.model.save(
                    f'models/{self.agent.MODEL_NAME}__{max_reward:>7.2f}max_'
                    f'{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

            # Handle epsilon
            if not self.agent.decay_epsilon() and reset_epsilon_game == -1:  # Epsilon was already at lowest value
                reset_epsilon_game = game_number + self.CONTINUE_AFTER_EPSILON_TARGET
            if game_number == reset_epsilon_game:
                print("Resetting epsilon!")
                reset_epsilon_game = -1
                self.agent.epsilon = 1
            game_number += 1

        self.plot_rewards(aggr_ep_rewards)
        pygame.quit()
        self.plot_rewards(aggr_ep_rewards)

    def handle_keyboard(self, toggle_show, toggle_limit_fps):
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_d]:
            if not toggle_show:
                self.show_track_forced = not self.show_track_forced
                toggle_show = True
        else:
            toggle_show = False

        if pressed[pygame.K_l]:
            if not toggle_limit_fps:
                self.limit_fps = not self.limit_fps
                toggle_limit_fps = True
        else:
            toggle_limit_fps = False

        return toggle_show, toggle_limit_fps

    def draw_screen(self, game_number, show_time):
        if self.show_track_forced or not game_number % self.SHOW_EVERY:
            return True, show_time
        else:
            self.clock.tick()
            show_time += self.clock.get_time() / 1000
            if show_time > self.SHOW_FPS_EVERY:
                self.screen.fill(BLACK)
                rect = self.screen.blit(pygame.font.SysFont('Comic Sans MS', 25).render(
                    "Rendering in quick mode with {}fps".format(1000 / self.clock.get_time()),
                    False, WHITE), (10, 10))
                pygame.display.update(rect)
                show_time = 0
            return False, show_time

    def plot_rewards(self, aggr_ep_rewards):
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
        plt.legend(loc=4)
        plt.savefig("models/{}-{}.png".format(self.agent.MODEL_NAME, int(time.time())))
        plt.show()

    def fill_agent_memory(self, dt, toggle_show, toggle_limit_fps):
        step = 0
        while step < self.agent.REPLAY_MEMORY_SIZE:
            self.env.init()
            done = False
            current_state = self.env.build_state(self.env.cars[0])

            while not done and step < self.agent.REPLAY_MEMORY_SIZE:  # Game loop
                pygame.event.get()  # Prevents OS from seeing game as not responding
                toggle_show, toggle_limit_fps = self.handle_keyboard(toggle_show, toggle_limit_fps)
                action = np.random.randint(0, self.env.ALLOWED_INPUTS)
                new_states, rewards, done = self.env.step([action], dt, step % 500 == 0)
                self.agent.update_replay_memory((current_state, action, rewards[0], new_states[0], done))
                current_state = new_states[0]
                step += 1

                if not step % 500:
                    print("Memory filled with {} steps".format(step))


# For repeatable results
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))  # Log which device is being used for Deep learning

# Render window in correct position on screen
os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"

# Create folder for models
if not os.path.isdir('models'):
    os.makedirs('models')

QTrainer(DrAIve(1, pygame.display.set_mode((QTrainer.DISPLAY_WIDTH, QTrainer.DISPLAY_HEIGHT))),
         "DrAIve-Prioritised-Targeted-DQN-TrainEvery4", False, False).run(None, None, 2500)
