import pygame
import numpy as np
import os

from pygame.math import Vector2
from math import sin, cos, tan, radians, degrees, copysign, floor

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)


class Car:
    def __init__(self, x, y, scale, angle=0.0, length=4, max_steering=50, max_acceleration=40.0,
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


def input_to_car(car, choice, dt):
    if choice in (0, 1, 2):
        if car.velocity.x < 0:
            car.acceleration = car.brake_deceleration
        else:
            car.acceleration += car.acceleration_speed * dt
    elif choice in (6, 7, 8):
        if car.velocity.x > 0:
            car.acceleration = -car.brake_deceleration
        else:
            car.acceleration -= car.acceleration_speed * dt
    else:
        car.acceleration = -copysign(car.drag, car.velocity.x)

    if choice in (0, 3, 6):
        if car.steering < 0:
            car.steering = 0
        car.steering += car.steering_speed * dt
    elif choice in (2, 5, 8):
        if car.steering > 0:
            car.steering = 0
        car.steering -= car.steering_speed * dt
    else:
        car.steering = 0

    # Clamp values
    car.acceleration = max(-car.max_acceleration, min(car.acceleration, car.max_acceleration))
    car.steering = max(-car.max_steering, min(car.steering, car.max_steering))


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


def get_track_lines():
    lines = []
    lines.append(((100, 100), (700, 700)))
    return lines


def calc_intersect(line1, line2):
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]
    if (x4 - x3) * (y1 - y2) - (x1 - x2) * (y4 - y3) == 0:
        return None
    t1 = ((y3 - y4) * (x1 - x3) + (x4 - x3) * (y1 - y3))/((x4 - x3) * (y1 - y2) - (x1 - x2) * (y4 - y3))
    t2 = ((y1 - y2) * (x1 - x3) + (x2 - x1) * (y1 - y3))/((x4 - x3) * (y1 - y2) - (x1 - x2) * (y4 - y3))

    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        return line1[0] + np.multiply(t1, tuple(np.subtract(line1[1], line1[0])))
    else:
        return None


def vision_lines(car, ppu, car_length, car_width):
    line_length = 400
    lines = []
    front_offset = (cos(radians(-car.angle)) * car_length / 2, sin(radians(-car.angle)) * car_length / 2)
    front_line = (np.multiply(car.position, ppu) + front_offset,
                  np.multiply(car.position, ppu) + (cos(radians(-car.angle))*line_length,
                                                    sin(radians(-car.angle)) * line_length) + front_offset)
    rear_line = (np.multiply(car.position, ppu) - front_offset,
                 np.multiply(car.position, ppu) - (cos(radians(-car.angle))*line_length,
                                                   sin(radians(-car.angle)) * line_length) - front_offset)
    
    lines.append(front_line)
    lines.append(rear_line)
    return lines


class Game:
    DISPLAY_WIDTH = 1920
    DISPLAY_HEIGHT = 1080

    def __init__(self):
        pygame.init()
        pygame.display.set_caption('DrAIve')
        self.screen = pygame.display.set_mode((self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False

    def run(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "car.png")
        car_image = pygame.image.load(image_path)
        car_scale = 0.5

        new_car_image_size = (floor(car_image.get_rect().size[0] * car_scale),
                              floor(car_image.get_rect().size[1] * car_scale))
        car_image = pygame.transform.scale(car_image, new_car_image_size)
        car = Car(0, 0, car_scale)
        ppu = 32 / car_scale

        track_lines = get_track_lines()

        while not self.exit:
            dt = self.clock.get_time() / 1000

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True
                print(event)

            pressed = pygame.key.get_pressed()

            input_to_car(car, keys_to_choice(pressed), dt)
            car.update(dt)

            self.screen.fill(BLACK)
            rotated = pygame.transform.rotate(car_image, car.angle)
            rect = rotated.get_rect()
            self.screen.blit(rotated, car.position * ppu - (rect.width / 2, rect.height / 2))
            for line in track_lines:
                pygame.draw.line(self.screen, WHITE, line[0], line[1], 5)
            for line in vision_lines(car, ppu, max(rect.width, rect.height), min(rect.width, rect.height)):
                pygame.draw.line(self.screen, WHITE, line[0], line[1], 2)
            print(calc_intersect(track_lines[0], vision_lines(car, ppu, rect.width, rect.height)[0]))
            pygame.display.update()
            self.clock.tick(self.ticks)
        pygame.quit()


game = Game()
game.run()
