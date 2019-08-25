import pygame
import matplotlib.pyplot as plt
import time
import random
import numpy as np
import tensorflow as tf
import os


class AgentTrainer:
    AGGREGATE_STATS_EVERY = 50  # episodes
    TICKS = 30  # Number of steps per second the AI is asked to give an action

    def __init__(self, env):
        self.env = env
        self.show_track_forced = True
        self.limit_fps = False

        # For repeatable results
        random.seed(1)
        np.random.seed(1)
        tf.set_random_seed(1)

        # Render window in correct position on screen
        os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"

        # Create folder for models
        if not os.path.isdir('models'):
            os.makedirs('models')

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

    @staticmethod
    def update_aggregate_rewards(game_number, game_rewards, aggr_ep_rewards):
        average_reward = sum(game_rewards[-AgentTrainer.AGGREGATE_STATS_EVERY:]) \
                         / len(game_rewards[-AgentTrainer.AGGREGATE_STATS_EVERY:])
        min_reward = min(game_rewards[-AgentTrainer.AGGREGATE_STATS_EVERY:])
        max_reward = max(game_rewards[-AgentTrainer.AGGREGATE_STATS_EVERY:])
        aggr_ep_rewards['ep'].append(game_number)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min_reward)
        aggr_ep_rewards['max'].append(max_reward)

        return average_reward, min_reward, max_reward

    @staticmethod
    def plot_rewards(aggr_ep_rewards, model_name):
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
        plt.legend(loc=4)
        plt.savefig("models/{}-{}.png".format(model_name, int(time.time())))
        plt.show()
