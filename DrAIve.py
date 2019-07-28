import pygame
import numpy as np
import os

from pygame.math import Vector2
from math import sin, cos, tan, radians, degrees, copysign, floor, atan2
from functools import reduce

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

VISION_LINE_LENGTH = 400

DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080


class Car:
    def __init__(self, x, y, scale, ppu, image, angle=0.0, length=4, max_steering=50, max_acceleration=40.0,
                 acceleration_speed=20.0, steering_speed=80, brake_deceleration=40.0, top_speed=20, drag=5):
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
                self.acceleration += self.acceleration_speed * dt
        elif choice in (6, 7, 8):
            if self.velocity.x > 0:
                self.acceleration = -self.brake_deceleration
            else:
                self.acceleration -= self.acceleration_speed * dt
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
                self.get_relative_point((0, 0)) + (cos(radians(-self.angle + angle)) * VISION_LINE_LENGTH,
                                                   sin(radians(-self.angle + angle)) * VISION_LINE_LENGTH) + offset)

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
                distances.append(calc_distance(line[0], intersection) / VISION_LINE_LENGTH)
            else:
                distances.append(1)
        return distances


class Track:
    def __init__(self):
        self.lines = []
        self.car_start_location = None
        self.car_start_angle = None

    def get_lines(self):
        return self.lines

    def add_line(self, line):
        self.lines.append(line)

    def generate_track_lines(self):
        self.lines.append(((100, 100), (700, 700)))
        self.lines.append(((100, 200), (700, 800)))

    def set_car_start(self, location, angle):
        self.car_start_location = location
        self.car_start_angle = angle


class Game:
    def __init__(self, current_track):
        pygame.init()
        pygame.display.set_caption('DrAIve')
        self.screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False
        self.track = current_track

    def run(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "car.png")
        car_image = pygame.image.load(image_path)
        car_scale = 0.3

        new_car_image_size = (floor(car_image.get_rect().size[0] * car_scale),
                              floor(car_image.get_rect().size[1] * car_scale))
        car_image = pygame.transform.scale(car_image, new_car_image_size)
        ppu = 32 / car_scale
        car = Car(self.track.car_start_location[0] / ppu, self.track.car_start_location[1] / ppu,
                  car_scale, ppu, car_image, self.track.car_start_angle)

        while not self.exit:
            dt = self.clock.get_time() / 1000

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True

            pressed = pygame.key.get_pressed()

            car.send_input(keys_to_choice(pressed), dt)
            car.update(dt)

            self.screen.fill(BLACK)

            for line in self.track.get_lines():
                pygame.draw.line(self.screen, WHITE, line[0], line[1], 3)
            for line in car.vision_lines():
                pygame.draw.line(self.screen, WHITE, line[0], line[1])
                vision_point = get_closest_vision_intersection(line, self.track)
                if vision_point is not None:
                    integer_point = (int(vision_point[0]), int(vision_point[1]))
                    pygame.draw.circle(self.screen, RED, integer_point, 8)
            for line in car.hitbox_lines():
                pygame.draw.line(self.screen, WHITE, line[0], line[1])

            rotated = pygame.transform.rotate(car_image, car.angle)
            rect = rotated.get_rect()
            self.screen.blit(rotated, car.position * ppu - (rect.width / 2, rect.height / 2))
            pygame.display.update()
            self.clock.tick(self.ticks)
        pygame.quit()


class TrackEditor:
    def __init__(self, edit_track):
        pygame.init()
        pygame.display.set_caption('Level Editor')
        self.screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False
        self.track = edit_track
        self.direction_arrow_size = 80

    def run(self):
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
                    stage += 1
                    move_next = False
            else:
                move_next = True

            self.screen.fill(BLACK)
            for line in self.track.get_lines():
                pygame.draw.line(self.screen, WHITE, line[0], line[1], 5)
            if current_line is not None:
                pygame.draw.line(self.screen, WHITE, current_line[0], current_line[1], 5)

            if stage == 0:
                current_line, first_line = self.add_lines(current_line, first_line, ev, pressed)
            elif stage == 1:
                stage += self.set_start_point_and_angle(ev)
            elif stage >= 2:
                if pressed[pygame.K_q]:
                    self.exit = True

            self.draw_start()

            pygame.display.update()
            self.clock.tick(self.ticks)
        pygame.quit()

    def add_lines(self, current_line, first_line, events, keys_pressed):
        if current_line is not None:
            current_line = (current_line[0], pygame.mouse.get_pos())
        for event in events:
            if event.type == pygame.MOUSEBUTTONUP:
                if current_line is None:
                    current_line = (pygame.mouse.get_pos(), pygame.mouse.get_pos())
                    first_line = current_line
                else:
                    self.track.add_line((current_line[0], pygame.mouse.get_pos()))
                    current_line = (pygame.mouse.get_pos(), pygame.mouse.get_pos())

        if keys_pressed[pygame.K_f] and current_line is not None:
            self.track.add_line((current_line[0], first_line[0]))
            current_line = None
            first_line = None

        return current_line, first_line

    def set_start_point_and_angle(self, events):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                print("Setting start location")
                self.track.car_start_location = pygame.mouse.get_pos()
            elif event.type == pygame.MOUSEBUTTONUP:
                self.track.car_start_angle = -angle_between(self.track.car_start_location, pygame.mouse.get_pos())
                return 1
        return 0

    def draw_start(self):
        if self.track.car_start_location is not None:
            pygame.draw.circle(self.screen, RED, self.track.car_start_location, 8)
        if self.track.car_start_angle is not None:
            pygame.draw.line(self.screen, WHITE, self.track.car_start_location,
                             np.add(self.track.car_start_location,
                                    (cos(radians(self.track.car_start_angle)) * self.direction_arrow_size,
                                     -sin(radians(self.track.car_start_angle)) * self.direction_arrow_size)), 5)


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
    for track_line in vision_track.get_lines():
        collision_location = calc_intersect(vision_line, track_line)
        if collision_location is not None:
            collisions.append(collision_location)
    if len(collisions) == 0:
        return None
    return reduce((lambda x, y: min_distance_point(vision_line, x, y)), collisions)


def car_hit_wall(car, current_track):
    for hit_line in car.hitbox_lines():
        for track_line in current_track.get_lines():
            if calc_intersect(hit_line, track_line) is not None:
                return True
    return False


def angle_between(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return degrees(atan2(dy, dx))


track = Track()
track_editor = TrackEditor(track)
track_editor.run()
game = Game(track)
game.run()
