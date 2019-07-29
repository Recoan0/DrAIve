import pygame
import tensorflow as tf
import numpy as np
import random
import time
import os

from pygame.math import Vector2
from math import sin, cos, tan, radians, degrees, copysign, floor, atan2
from functools import reduce
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


class Car:
    VISION_LINE_LENGTH = 400

    def __init__(self, x, y, scale, ppu, image, angle=0.0, length=4, max_steering=45, max_acceleration=40.0,
                 acceleration_speed=15.0, steering_speed=60, brake_deceleration=20.0, top_speed=8, drag=5):
        self.position = Vector2(x, y)
        self.velocity = Vector2(0.0, 0.0)
        self.angle = angle
        self.length = length * scale
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        self.acceleration_speed = acceleration_speed
        self.steering_speed = steering_speed
        self.brake_deceleration = brake_deceleration
        self.top_speed = top_speed
        self.drag = drag
        self.ppu = ppu
        self.image = image
        self.width = pygame.transform.rotate(self.image, 0).get_rect().width
        self.height = pygame.transform.rotate(self.image, 0).get_rect().height

        self.acceleration = 0.0
        self.steering = 0.0
        self.current_reward_gate = 0

    def update(self, dt):
        self.velocity += (self.acceleration * dt, 0)

        # Clamp velocity
        self.velocity.x = max(-self.top_speed, min(self.velocity.x, self.top_speed))

        if self.steering:
            turning_radius = self.length / tan(radians(self.steering))
            angular_velocity = self.velocity.x / turning_radius
        else:
            angular_velocity = 0

        self.position += self.velocity.rotate(-self.angle) * dt
        self.angle += degrees(angular_velocity) * dt

        self.position.x = max(0, min(28, self.position.x))
        self.position.y = max(0, min(14, self.position.y))

    def send_input(self, choice, dt):
        if choice in (0, 1, 2):
            if self.velocity.x < 0:
                self.acceleration = self.brake_deceleration
            else:
                self.acceleration = self.acceleration_speed
        elif choice in (6, 7, 8):
            if self.velocity.x > 0:
                self.acceleration = -self.brake_deceleration
            else:
                self.acceleration = -self.acceleration_speed * 0.5
        else:
            self.acceleration = -copysign(self.drag, self.velocity.x)

        if choice in (0, 3, 6):
            if self.steering < 0:
                self.steering = 0
            self.steering += self.steering_speed * dt
        elif choice in (2, 5, 8):
            if self.steering > 0:
                self.steering = 0
            self.steering -= self.steering_speed * dt
        else:
            self.steering = 0

        # Clamp values
        self.acceleration = max(-self.max_acceleration, min(self.acceleration, self.max_acceleration))
        self.steering = max(-self.max_steering, min(self.steering, self.max_steering))

    def get_offsets(self):
        # Returns front_offset, side_offset
        return ((cos(radians(-self.angle)) * self.height, sin(radians(-self.angle)) * self.height),
                (-sin(radians(-self.angle)) * self.width / 4, cos(radians(-self.angle)) * self.width / 4))

    def get_relative_point(self, offset):
        return np.multiply(self.position, self.ppu) + offset

    def get_vision_line(self, offset, angle):
        return (self.get_relative_point(offset),
                self.get_relative_point((0, 0)) + (cos(radians(-self.angle + angle)) * self.VISION_LINE_LENGTH,
                                                   sin(radians(
                                                       -self.angle + angle)) * self.VISION_LINE_LENGTH) + offset)

    def hitbox_lines(self):
        # Returns hitbox lines in clockwise direction
        front_offset, side_offset = self.get_offsets()
        front_line = (self.get_relative_point(np.add(front_offset, side_offset)),
                      self.get_relative_point(np.subtract(front_offset, side_offset)))
        right_line = (self.get_relative_point(np.add(front_offset, side_offset)),
                      self.get_relative_point(np.subtract(side_offset, front_offset)))
        rear_line = (self.get_relative_point(np.subtract(side_offset, front_offset)),
                     self.get_relative_point(np.subtract(np.negative(front_offset), side_offset)))
        left_line = (self.get_relative_point(np.subtract(front_offset, side_offset)),
                     self.get_relative_point(np.subtract(np.negative(side_offset), front_offset)))

        return front_line, right_line, rear_line, left_line

    def vision_lines(self):
        # After training AI order can not change!
        lines = []
        front_offset, side_offset = self.get_offsets()

        front_line = self.get_vision_line(front_offset, 0)
        rear_line = self.get_vision_line(np.negative(front_offset), 180)
        front_right_diagonal_side_line = self.get_vision_line(np.add(front_offset, side_offset), 45)
        front_left_diagonal_side_line = self.get_vision_line(np.subtract(front_offset, side_offset), -45)
        front_right_perpendicular_side_line = self.get_vision_line(np.add(front_offset, side_offset), 90)
        front_left_perpendicular_side_line = self.get_vision_line(np.subtract(front_offset, side_offset), -90)
        front_right_forward_side_line = self.get_vision_line(np.add(front_offset, side_offset), 10)
        front_left_forward_side_line = self.get_vision_line(np.subtract(front_offset, side_offset), -10)
        rear_right_side_line = self.get_vision_line(np.subtract(side_offset, front_offset), 135)
        rear_left_side_line = self.get_vision_line(np.subtract(np.negative(side_offset), front_offset), -135)

        lines.append(front_line)
        lines.append(front_right_diagonal_side_line)
        lines.append(front_left_diagonal_side_line)
        lines.append(front_right_perpendicular_side_line)
        lines.append(front_left_perpendicular_side_line)
        lines.append(front_right_forward_side_line)
        lines.append(front_left_forward_side_line)
        lines.append(rear_line)
        lines.append(rear_right_side_line)
        lines.append(rear_left_side_line)
        return lines

    def get_vision_distances(self, vision_track):
        distances = []
        for line in self.vision_lines():
            intersection = get_closest_vision_intersection(line, vision_track)
            if intersection is not None:
                # Distance normalized for better AI learning
                distances.append(calc_distance(line[0], intersection) / self.VISION_LINE_LENGTH)
            else:
                distances.append(1)
        return distances

    def hit_wall(self, current_track):
        for hit_line in self.hitbox_lines():
            for track_line in current_track.get_walls():
                if calc_intersect(hit_line, track_line) is not None:
                    return True
        return False

    def hit_reward_gate(self, current_track):
        for hit_line in self.hitbox_lines():
            if self.current_reward_gate < len(current_track.get_reward_gates()) and \
                    calc_intersect(hit_line, current_track.get_reward_gates()[self.current_reward_gate]) is not None:
                return True
        return False


class Track:
    def __init__(self):
        self.walls = []
        self.reward_gates = []
        self.car_start_location = (0, 0)
        self.car_start_angle = 0

    def get_walls(self):
        return self.walls

    def get_reward_gates(self):
        return self.reward_gates

    def add_wall(self, line):
        self.walls.append(line)

    def add_reward_gate(self, gate):
        self.reward_gates.append(gate)

    def set_car_start(self, location, angle):
        self.car_start_location = location
        self.car_start_angle = angle

    def __str__(self):
        return_string = "walls: "
        for wall in self.get_walls():
            return_string += wall.__str__() + " "
        return_string += "\ngates: "
        for gate in self.get_reward_gates():
            return_string += gate.__str__() + " "
        return_string += "\nstart_location: " + self.car_start_location.__str__()
        return_string += "\nstart_angle: " + self.car_start_angle.__str__()
        return return_string


class TrackEditor:
    TICKS = 60

    def __init__(self, screen):
        pygame.display.set_caption('Level Editor')
        self.screen = screen
        self.exit = False
        self.direction_arrow_size = 80

    def run(self, clock):
        edit_track = Track()
        stage = 0
        current_line = None
        first_line = None
        move_next = True

        while not self.exit:
            ev = pygame.event.get()
            for event in ev:
                if event.type == pygame.QUIT:
                    self.exit = True

            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_n]:
                if move_next:
                    if stage == 2:  # Add finish line to track
                        edit_track.add_reward_gate(edit_track.get_reward_gates()[0])
                    stage += 1
                    move_next = False
            else:
                move_next = True

            self.screen.fill(BLACK)
            for line in edit_track.get_walls():
                pygame.draw.line(self.screen, WHITE, line[0], line[1], 5)
            for line in edit_track.get_reward_gates():
                pygame.draw.line(self.screen, GREEN, line[0], line[1], 3)

            if stage == 0:
                current_line, first_line = self.add_walls(current_line, first_line, ev, pressed, edit_track)
                if current_line is not None:
                    pygame.draw.line(self.screen, WHITE, current_line[0], current_line[1], 5)
            elif stage == 1:
                stage += self.set_start_point_and_angle(ev, edit_track)
            elif stage == 2:
                current_line = self.add_reward_gates(current_line, ev, edit_track)
                if current_line is not None:
                    pygame.draw.line(self.screen, GREEN, current_line[0], current_line[1], 3)
            elif stage >= 3:
                if pressed[pygame.K_q]:
                    self.exit = True

            self.draw_start(edit_track)

            pygame.display.update()
            clock.tick(self.TICKS)
        return edit_track

    @staticmethod
    def add_walls(current_line, first_line, events, keys_pressed, edit_track):
        if current_line is not None:
            current_line = (current_line[0], pygame.mouse.get_pos())
        for event in events:
            if event.type == pygame.MOUSEBUTTONUP:
                if current_line is None:
                    current_line = (pygame.mouse.get_pos(), pygame.mouse.get_pos())
                    first_line = current_line
                else:
                    edit_track.add_wall((current_line[0], pygame.mouse.get_pos()))
                    current_line = (pygame.mouse.get_pos(), pygame.mouse.get_pos())

        if keys_pressed[pygame.K_f] and current_line is not None:
            edit_track.add_wall((current_line[0], first_line[0]))
            current_line = None
            first_line = None

        return current_line, first_line

    @staticmethod
    def set_start_point_and_angle(events, edit_track):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                print("Setting start location")
                edit_track.car_start_location = pygame.mouse.get_pos()
                print(edit_track.car_start_location)
            elif event.type == pygame.MOUSEBUTTONUP:
                edit_track.car_start_angle = -angle_between_points(edit_track.car_start_location,
                                                                   pygame.mouse.get_pos())
                return 1
        return 0

    @staticmethod
    def add_reward_gates(current_line, events, edit_track):
        if current_line is not None:
            current_line = (current_line[0], pygame.mouse.get_pos())
        for event in events:
            if event.type == pygame.MOUSEBUTTONUP:
                if current_line is None:
                    current_line = (pygame.mouse.get_pos(), pygame.mouse.get_pos())
                else:
                    edit_track.add_reward_gate((current_line[0], pygame.mouse.get_pos()))
                    current_line = None

        return current_line

    def draw_start(self, edit_track):
        if edit_track.car_start_location is not None:
            pygame.draw.circle(self.screen, RED, edit_track.car_start_location, 8)
        if edit_track.car_start_angle is not None:
            pygame.draw.line(self.screen, WHITE, edit_track.car_start_location,
                             np.add(edit_track.car_start_location,
                                    (cos(radians(edit_track.car_start_angle)) * self.direction_arrow_size,
                                     -sin(radians(edit_track.car_start_angle)) * self.direction_arrow_size)), 5)


class Game:
    ACTION_SPACE_SIZE = 9  # 9 different input possibilities
    TICKS = 60

    def __init__(self, screen, gate_reward, finish_reward, crash_punishment, fuel_cost):
        self.screen = screen
        self.keep_going = True
        self.gate_reward = gate_reward
        self.finish_reward = finish_reward
        self.crash_punishment = crash_punishment
        self.fuel_cost = fuel_cost

    def run(self, agent, clock, current_track, show_screen):
        toggled = False
        show_screen_every = 5000
        show_time = 0

        exit_game = False
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "car.png")
        car_image = pygame.image.load(image_path)
        car_scale = 0.3

        new_car_image_size = (floor(car_image.get_rect().size[0] * car_scale),
                              floor(car_image.get_rect().size[1] * car_scale))
        car_image = pygame.transform.scale(car_image, new_car_image_size)
        ppu = 32 / car_scale
        car = Car(current_track.car_start_location[0] / ppu, current_track.car_start_location[1] / ppu,
                  car_scale, ppu, car_image, current_track.car_start_angle)

        current_state = self.build_state(car, current_track)
        game_score = 0

        while not exit_game:
            if show_screen:
                dt = clock.get_time() / 1000
            else:
                dt = 1 / self.TICKS  # Standard delta time for running more fps but setting movement to same delta

            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_d]:
                if not toggled:
                    show_screen = not show_screen
                    toggled = True
            else:
                toggled = False

            if agent is not None:
                ### AI ###
                if np.random.random() > agent.epsilon:
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    action = np.random.randint(0, self.ACTION_SPACE_SIZE)
                car.send_input(action, dt)
                ##########
            else:
                ### PLAYER ###
                car.send_input(keys_to_choice(pressed), dt)
                ##############

            car.update(dt)
            reward, exit_game = self.rate_action(car, current_track)

            if agent is not None:
                ### AI ###
                game_score += reward
                new_state = self.build_state(car, current_track)
                agent.update_replay_memory((current_state, action, reward, new_state, exit_game))
                agent.train(exit_game)
                current_state = new_state
                ##########

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit_game = True
                    self.keep_going = False

            if show_screen:
                self.draw_screen(car, current_track, car_image, ppu)
                pygame.display.update()
                clock.tick(self.TICKS)
            else:
                clock.tick()
                show_time += clock.get_time()
                if show_time > show_screen_every:
                    self.screen.fill(BLACK)
                    rect = self.screen.blit(pygame.font.SysFont('Comic Sans MS', 25)
                                            .render(f"Rendering in quick mode with {1000 / clock.get_time()}fps",
                                                    False, WHITE), (10, 10))
                    pygame.display.update(rect)
                    print(f"{1000 / clock.get_time()}fps")
                    show_time = 0
        return self.keep_going, game_score

    def rate_action(self, car, current_track):  # Returns score and whether the level should be reset
        if car.hit_wall(current_track):
            return self.crash_punishment, True
        elif car.hit_reward_gate(current_track):
            car.current_reward_gate += 1
            if car.current_reward_gate % len(current_track.get_reward_gates()) == 0:
                return self.finish_reward, True
            return self.gate_reward, False
        else:
            return self.fuel_cost, False

    def draw_screen(self, car, current_track, car_image, ppu):
        self.screen.fill(BLACK)
        for line in current_track.get_walls():
            pygame.draw.line(self.screen, WHITE, line[0], line[1], 3)
        for line in current_track.get_reward_gates():
            pygame.draw.line(self.screen, GREEN, line[0], line[1], 3)
        for line in car.vision_lines():
            pygame.draw.line(self.screen, WHITE, line[0], line[1])
            vision_point = get_closest_vision_intersection(line, current_track)
            if vision_point is not None:
                integer_point = (int(vision_point[0]), int(vision_point[1]))
                pygame.draw.circle(self.screen, RED, integer_point, 8)
        for line in car.hitbox_lines():
            pygame.draw.line(self.screen, WHITE, line[0], line[1])
        rotated = pygame.transform.rotate(car_image, car.angle)
        rect = rotated.get_rect()
        self.screen.blit(rotated, car.position * ppu - (rect.width / 2, rect.height / 2))

    @staticmethod
    def build_state(car, track):
        state = () + tuple(car.get_vision_distances(track)) + \
                (normalize_angle(car.angle),) + tuple(component / car.top_speed for component in car.velocity)
        return state


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQNAgent:
    REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
    MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in memory to start training
    MODEL_NAME = "DAIv0"
    MINIBATCH_SIZE = 64  # How many steps/samples to use for training
    DISCOUNT = 0.99
    UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
    MEMORY_FRACTION = 0.20

    # Exploration settings
    EPSILON_DECAY = 0.99975
    MIN_EPSILON = 0.001

    def __init__(self):
        # main model, gets trained every step
        self.model = self.create_model()

        # Target model, this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.tensor_board = ModifiedTensorBoard(log_dir=f"logs/{self.MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

        self.epsilon = 1

    @staticmethod
    def create_model():
        model = Sequential()
        model.add(Dense(20, input_shape=(13,)))
        model.add(Activation("relu"))
        model.add(Dense(20))
        model.add(Activation("relu"))
        model.add(Dense(9, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *(13,)))[0]

    def decay_epsilon(self):
        if self.epsilon > self.MIN_EPSILON:
            self.epsilon *= self.EPSILON_DECAY
            self.epsilon = max(self.MIN_EPSILON, self.epsilon)

    def train(self, terminal_state):
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X), np.array(y), batch_size=self.MINIBATCH_SIZE, verbose=0,
                       shuffle=False, callbacks=[self.tensor_board] if terminal_state else None)

        # updating to determine if we want to update target_model yet
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


class QTrainer:
    TRACK_AMOUNT = 5

    # These need to remain the same after AI has started training
    GATE_REWARD = 100
    FINISH_REWARD = 1000
    CRASH_PUNISHMENT = -1000
    FUEL_COST = 0
    MIN_REWARD = -1000

    DISPLAY_WIDTH = 1920
    DISPLAY_HEIGHT = 1080
    SHOW_EVERY = 25

    EPISODES = 5000  # amount of games, currently not used
    AGGREGATE_STATS_EVERY = 50  # episodes

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
        self.clock = pygame.time.Clock()
        self.tracks = []
        self.agent = DQNAgent()

    def run(self):
        game_number = 1
        game_rewards = []
        self.generate_tracks(self.TRACK_AMOUNT)

        track_string = ""
        for track in self.tracks:
            track_string += track.__str__() + "\n\n"
        with open(f"Tracks_{int(time.time())}.txt", "w") as text_file:
            print(track_string, file=text_file)

        pygame.display.set_caption('DrAIve')
        game = Game(self.screen, self.GATE_REWARD, self.FINISH_REWARD, self.CRASH_PUNISHMENT, self.FUEL_COST)

        run = True
        while run:
            self.agent.tensor_board.step = game_number
            run, game_reward = game.run(self.agent, self.clock, random.choice(self.tracks),
                                        game_number % self.SHOW_EVERY == 1)
            game_number += 1

            # Append game reward to a list and log stats (every given number of games)
            game_rewards.append(game_reward)
            if not game_number % self.AGGREGATE_STATS_EVERY or game_number == 1:
                average_reward = sum(game_rewards[-self.AGGREGATE_STATS_EVERY:]) \
                                 / len(game_rewards[-self.AGGREGATE_STATS_EVERY:])
                min_reward = min(game_rewards[-self.AGGREGATE_STATS_EVERY:])
                max_reward = max(game_rewards[-self.AGGREGATE_STATS_EVERY:])
                self.agent.tensor_board.update_stats(reward_avg=average_reward, reward_min=min_reward,
                                                     reward_max=max_reward,
                                                     epsilon=self.agent.epsilon)

                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= self.MIN_REWARD:
                    self.agent.model.save(
                        f'models/{self.agent.MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

            # Decay epsilon
            self.agent.decay_epsilon()

        pygame.quit()

    def generate_tracks(self, amount):
        for i in range(amount):
            self.tracks.append(TrackEditor(self.screen).run(self.clock))


class ManualPlayer:
    DISPLAY_WIDTH = 1920
    DISPLAY_HEIGHT = 1080

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
        self.clock = pygame.time.Clock()
        self.tracks = []

    def run(self, make_track):
        if make_track:
            track = TrackEditor(self.screen).run(self.clock)
        else:
            track = Track()
        Game(self.screen, 0, 0, 0, 0).run(None, self.clock, track, True)


def keys_to_choice(pressed):
    if pressed[pygame.K_UP]:
        if pressed[pygame.K_LEFT]:
            return 0
        elif pressed[pygame.K_RIGHT]:
            return 2
        else:
            return 1
    elif pressed[pygame.K_DOWN]:
        if pressed[pygame.K_LEFT]:
            return 6
        elif pressed[pygame.K_RIGHT]:
            return 8
        else:
            return 7
    else:
        if pressed[pygame.K_LEFT]:
            return 3
        elif pressed[pygame.K_RIGHT]:
            return 5
        else:
            return 4


def calc_distance(point1, point2):
    return np.linalg.norm(np.subtract(point1, point2))


def min_distance_point(line, point1, point2):
    origin_point = line[0]
    if calc_distance(origin_point, point1) < calc_distance(origin_point, point2):
        return point1
    return point2


def calc_intersect(line1, line2):
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]
    if (x4 - x3) * (y1 - y2) - (x1 - x2) * (y4 - y3) == 0:
        return None
    t1 = ((y3 - y4) * (x1 - x3) + (x4 - x3) * (y1 - y3)) / ((x4 - x3) * (y1 - y2) - (x1 - x2) * (y4 - y3))
    t2 = ((y1 - y2) * (x1 - x3) + (x2 - x1) * (y1 - y3)) / ((x4 - x3) * (y1 - y2) - (x1 - x2) * (y4 - y3))

    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        return line1[0] + np.multiply(t1, tuple(np.subtract(line1[1], line1[0])))
    else:
        return None


def get_closest_vision_intersection(vision_line, vision_track):
    collisions = []
    for track_line in vision_track.get_walls():
        collision_location = calc_intersect(vision_line, track_line)
        if collision_location is not None:
            collisions.append(collision_location)
    if len(collisions) == 0:
        return None
    return reduce((lambda x, y: min_distance_point(vision_line, x, y)), collisions)


def angle_between_points(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return degrees(atan2(dy, dx))


def normalize_angle(angle):
    return (angle % 360) / 360


# For repeatable results
random.seed(1)
np.random.seed(1)
tf.compat.v1.set_random_seed(1)

# Create folder for models
if not os.path.isdir('models'):
    os.makedirs('models')

QTrainer().run()  # Run with AI!
# ManualPlayer().run(False)  # Run with manual play!