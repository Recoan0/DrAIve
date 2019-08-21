import pygame
import numpy as np
import os
import time

from pygame.math import Vector2
from math import sin, cos, radians, degrees, floor, atan2
from functools import reduce


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


class Car:
    VISION_LINE_LENGTH = 400

    def __init__(self, x, y, scale, ppu, image, angle=0.0, length=4, acceleration_speed=10.0,
                 steering_speed=4, top_speed=4, drag=0.97, grip=0.1, convert_efficiency=0.1):
        self.position = Vector2(x, y)
        self.velocity = Vector2(0.0, 0.0)
        self.angle = angle
        self.length = length * scale
        self.acceleration_speed = acceleration_speed
        self.steering_speed = steering_speed
        self.top_speed = top_speed
        self.drag = drag
        self.grip = grip
        self.convert_efficiency = convert_efficiency
        self.ppu = ppu
        self.image = image
        self.width = pygame.transform.rotate(self.image, 0).get_rect().width
        self.height = pygame.transform.rotate(self.image, 0).get_rect().height

        self.acceleration = 0.0
        self.current_reward_gate = 0
        self.wrong_reward_gate = -1

    def update(self, dt):
        self.velocity.x += cos(radians(self.angle)) * self.acceleration * dt
        self.velocity.y += -sin(radians(self.angle)) * self.acceleration * dt

        # Clamp velocity
        self.velocity.x = max(-self.top_speed, min(self.velocity.x, self.top_speed))
        self.velocity.y = max(-self.top_speed, min(self.velocity.y, self.top_speed))

        self.position += (self.velocity.x * dt, self.velocity.y * dt)

        # Apply directional grip
        relative_velocity = self.velocity.rotate(self.angle)
        relative_angle = self.position.angle_to(relative_velocity)
        velocity_dy = relative_velocity.y * self.grip
        relative_velocity.y -= velocity_dy
        relative_velocity.x += cos(radians(relative_angle)) * velocity_dy * self.convert_efficiency
        relative_velocity.x *= self.drag
        self.velocity = relative_velocity.rotate(-self.angle)

        self.position.x = max(0, min(1920 / self.ppu, self.position.x))
        self.position.y = max(0, min(1080 / self.ppu, self.position.y))

    def send_input(self, choice):
        # choice = self.transform_actions(choice)
        if choice in (0, 1, 2):
            self.acceleration = self.acceleration_speed
        elif choice in (6, 7, 8):
            self.acceleration = -self.acceleration_speed * 0.6
        else:
            self.acceleration = 0

        if choice in (0, 3, 6):
            self.angle += self.steering_speed
        elif choice in (2, 5, 8):
            self.angle -= self.steering_speed

    @staticmethod
    def transform_actions(action):
        # From 6 to 9 actions, skip action 345
        if action > 2:
            action += 3
        return action

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
            intersection = Car.get_closest_vision_intersection(line, vision_track)
            if intersection is not None:
                # Distance normalized for better AI learning
                distances.append(Car.calc_distance(line[0], intersection) / self.VISION_LINE_LENGTH)
            else:
                distances.append(1)
        return distances

    def hit_wall(self, current_track):
        for hit_line in self.hitbox_lines():
            for track_line in current_track.get_walls():
                if Car.calc_intersect(hit_line, track_line) is not None:
                    return True
        return False

    # Do not call multiple times, reward_gate_number will be incremented each time
    def hit_right_reward_gate(self, current_track):
        for hit_line in self.hitbox_lines():
            if self.current_reward_gate < len(current_track.get_reward_gates()) and \
                    Car.calc_intersect(hit_line, current_track.get_reward_gates()[self.current_reward_gate]) is not None:
                gate_amount = len(current_track.get_reward_gates()) - 1
                self.wrong_reward_gate = (self.current_reward_gate - 1) % gate_amount
                self.current_reward_gate = (self.current_reward_gate + 1) % gate_amount
                return True
        return False

    # Can not place too many reward gates next to each other, if car hits multiple at a time will get punished
    def hit_wrong_reward_gate(self, current_track):
        self.wrong_reward_gate = self.wrong_reward_gate % (len(current_track.get_reward_gates()) - 1)
        for hit_line in self.hitbox_lines():
            if Car.calc_intersect(hit_line, current_track.get_reward_gates()[self.wrong_reward_gate]) is not None:
                self.wrong_reward_gate = (self.wrong_reward_gate - 1) % (len(current_track.get_reward_gates()) - 1)
                return True
        return False

    @staticmethod
    def calc_distance(point1, point2):
        return np.linalg.norm(np.subtract(point1, point2))

    @staticmethod
    def min_distance_point(line, point1, point2):
        origin_point = line[0]
        if Car.calc_distance(origin_point, point1) < Car.calc_distance(origin_point, point2):
            return point1
        return point2

    @staticmethod
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

    @staticmethod
    def get_closest_vision_intersection(vision_line, vision_track):
        collisions = []
        for track_line in vision_track.get_walls():
            collision_location = Car.calc_intersect(vision_line, track_line)
            if collision_location is not None:
                collisions.append(collision_location)
        if len(collisions) == 0:
            return None
        return reduce((lambda x, y: Car.min_distance_point(vision_line, x, y)), collisions)


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

    @staticmethod
    def set_standard_tracks():
        track1 = Track()
        track1.car_start_location = (183, 655)
        track1.car_start_angle = 93.9603118304584
        track1.walls = (((120, 765), (77, 336)), ((77, 336), (104, 193)), ((104, 193), (209, 107)), ((209, 107), (377, 68)), ((377, 68), (833, 44)), ((833, 44), (1380, 69)), ((1380, 69), (1517, 110)), ((1517, 110), (1595, 159)), ((1595, 159), (1651, 303)), ((1651, 303), (1629, 385)), ((1629, 385), (1501, 439)), ((1501, 439), (1267, 454)), ((1267, 454), (1094, 407)), ((1094, 407), (937, 359)), ((937, 359), (799, 387)), ((799, 387), (592, 484)), ((592, 484), (486, 579)), ((486, 579), (495, 651)), ((495, 651), (591, 689)), ((591, 689), (759, 682)), ((759, 682), (964, 689)), ((964, 689), (1192, 613)), ((1192, 613), (1406, 535)), ((1406, 535), (1673, 519)), ((1673, 519), (1830, 646)), ((1830, 646), (1849, 739)), ((1849, 739), (1855, 867)), ((1855, 867), (1756, 954)), ((1756, 954), (1537, 989)), ((1537, 989), (1124, 964)), ((1124, 964), (772, 954)), ((772, 954), (475, 962)), ((475, 962), (251, 923)), ((251, 923), (120, 765)), ((281, 723), (229, 346)), ((229, 346), (252, 273)), ((252, 273), (324, 227)), ((324, 227), (447, 180)), ((447, 180), (718, 174)), ((718, 174), (894, 160)), ((894, 160), (1279, 199)), ((1279, 199), (1346, 260)), ((1346, 260), (1274, 283)), ((1274, 283), (978, 212)), ((978, 212), (790, 208)), ((790, 208), (561, 288)), ((561, 288), (450, 416)), ((450, 416), (336, 544)), ((336, 544), (345, 710)), ((345, 710), (462, 755)), ((462, 755), (634, 799)), ((634, 799), (1019, 783)), ((1019, 783), (1439, 661)), ((1439, 661), (1626, 667)), ((1626, 667), (1673, 754)), ((1673, 754), (1615, 828)), ((1615, 828), (1397, 854)), ((1397, 854), (635, 834)), ((635, 834), (393, 788)), ((393, 788), (281, 723)))
        track1.reward_gates = (((258, 558), (100, 569)), ((229, 344), (79, 339)), ((315, 232), (208, 107)), ((537, 176), (528, 61)), ((905, 158), (914, 47)), ((1281, 199), (1381, 70)), ((1347, 262), (1636, 342)), ((1330, 447), (1270, 282)), ((938, 348), (980, 213)), ((796, 388), (565, 287)), ((484, 576), (338, 547)), ((588, 689), (630, 797)), ((964, 691), (1015, 783)), ((1405, 537), (1436, 659)), ((1672, 518), (1626, 665)), ((1844, 863), (1672, 751)), ((1758, 952), (1613, 829)), ((1251, 968), (1241, 852)), ((809, 953), (799, 840)), ((246, 922), (389, 788)), ((258, 558), (100, 569)))

        track2 = Track()
        track2.car_start_location = (667, 859)
        track2.car_start_angle = 1.4320961841646465
        track2.walls = (((860, 800), (1472, 767)), ((1472, 767), (1595, 720)), ((1595, 720), (1618, 579)), ((1618, 579), (1584, 355)), ((1584, 355), (1463, 261)), ((1463, 261), (1284, 303)), ((1284, 303), (1157, 427)), ((1157, 427), (1140, 564)), ((1140, 564), (1100, 668)), ((1100, 668), (955, 718)), ((955, 718), (792, 700)), ((792, 700), (720, 663)), ((720, 663), (604, 540)), ((604, 540), (573, 459)), ((573, 459), (535, 374)), ((535, 374), (420, 335)), ((420, 335), (297, 454)), ((297, 454), (287, 588)), ((287, 588), (309, 652)), ((309, 652), (351, 726)), ((351, 726), (422, 774)), ((422, 774), (663, 794)), ((663, 794), (860, 800)), ((408, 903), (811, 916)), ((811, 916), (1262, 909)), ((1262, 909), (1640, 904)), ((1640, 904), (1766, 814)), ((1766, 814), (1845, 693)), ((1845, 693), (1863, 517)), ((1863, 517), (1811, 317)), ((1811, 317), (1744, 193)), ((1744, 193), (1496, 105)), ((1496, 105), (1291, 123)), ((1291, 123), (1027, 230)), ((1027, 230), (968, 412)), ((968, 412), (922, 473)), ((922, 473), (802, 467)), ((802, 467), (729, 365)), ((729, 365), (643, 204)), ((643, 204), (408, 161)), ((408, 161), (182, 232)), ((182, 232), (43, 493)), ((43, 493), (44, 731)), ((44, 731), (224, 884)), ((224, 884), (408, 903)))
        track2.reward_gates = (((753, 796), (751, 912)), ((984, 793), (993, 913)), ((1207, 781), (1218, 907)), ((1470, 769), (1512, 903)), ((1591, 718), (1768, 813)), ((1620, 580), (1846, 685)), ((1584, 354), (1806, 313)), ((1465, 254), (1494, 104)), ((1280, 304), (1168, 176)), ((1156, 429), (983, 379)), ((953, 714), (890, 472)), ((600, 540), (802, 466)), ((541, 373), (671, 262)), ((424, 336), (411, 162)), ((367, 388), (185, 234)), ((290, 447), (85, 417)), ((281, 575), (45, 585)), ((308, 650), (150, 817)), ((420, 777), (383, 898)), ((592, 790), (574, 902)), ((753, 796), (751, 912)))

        track3 = Track()
        track3.car_start_location = (766, 782)
        track3.car_start_angle = -169.21570213243743
        track3.walls = (((825, 615), (757, 691)), ((757, 691), (576, 725)), ((576, 725), (389, 686)), ((389, 686), (337, 529)), ((337, 529), (423, 429)), ((423, 429), (569, 424)), ((569, 424), (757, 474)), ((757, 474), (825, 615)), ((1106, 620), (1220, 458)), ((1220, 458), (1413, 430)), ((1413, 430), (1521, 438)), ((1521, 438), (1598, 516)), ((1598, 516), (1544, 655)), ((1544, 655), (1353, 702)), ((1353, 702), (1106, 620)), ((265, 360), (376, 266)), ((376, 266), (557, 275)), ((557, 275), (765, 349)), ((765, 349), (867, 387)), ((867, 387), (940, 523)), ((940, 523), (1100, 355)), ((1100, 355), (1276, 284)), ((1276, 284), (1499, 283)), ((1499, 283), (1742, 368)), ((1742, 368), (1818, 511)), ((1818, 511), (1741, 680)), ((1741, 680), (1570, 839)), ((1570, 839), (1378, 859)), ((1378, 859), (1164, 837)), ((1164, 837), (989, 713)), ((989, 713), (885, 848)), ((885, 848), (621, 909)), ((621, 909), (302, 835)), ((302, 835), (168, 555)), ((168, 555), (265, 360)))
        track3.reward_gates = (((576, 728), (616, 903)), ((392, 689), (302, 833)), ((339, 528), (167, 555)), ((422, 427), (262, 362)), ((544, 424), (544, 276)), ((695, 456), (779, 358)), ((801, 564), (910, 475)), ((1023, 728), (1113, 626)), ((1244, 841), (1265, 675)), ((1402, 852), (1377, 699)), ((1569, 838), (1490, 666)), ((1759, 640), (1574, 581)), ((1771, 425), (1580, 493)), ((1513, 288), (1491, 434)), ((1281, 285), (1320, 437)), ((1102, 357), (1214, 459)), ((1029, 433), (1140, 571)), ((767, 680), (903, 819)), ((576, 728), (616, 903)))

        track4 = Track()
        track4.car_start_location = (1105, 308)
        track4.car_start_angle = -177.83308531681337
        track4.walls = (((533, 656), (458, 589)), ((458, 589), (463, 460)), ((463, 460), (574, 368)), ((574, 368), (1296, 369)), ((1296, 369), (1372, 457)), ((1372, 457), (1371, 564)), ((1371, 564), (1278, 658)), ((1278, 658), (533, 656)), ((471, 826), (281, 691)), ((281, 691), (276, 431)), ((276, 431), (479, 232)), ((479, 232), (1364, 229)), ((1364, 229), (1569, 402)), ((1569, 402), (1581, 584)), ((1581, 584), (1408, 811)), ((1408, 811), (471, 826)) )
        track4.reward_gates = (((1004, 231), (1006, 364)), ((810, 230), (807, 365)), ((603, 235), (632, 366)), ((428, 280), (529, 398)), ((280, 427), (459, 457)), ((276, 527), (457, 537)), ((277, 657), (456, 583)), ((399, 776), (530, 653)), ((590, 823), (598, 657)), ((792, 821), (795, 655)), ((945, 817), (945, 654)), ((1111, 814), (1104, 661)), ((1232, 812), (1225, 659)), ((1410, 804), (1297, 640)), ((1520, 665), (1370, 566)), ((1577, 504), (1376, 512)), ((1473, 321), (1344, 421)), ((1361, 231), (1296, 373)), ((1227, 232), (1228, 363)), ((1124, 226), (1127, 368)), ((1004, 231), (1006, 364)))

        track5 = Track()
        track5.car_start_location = (1579, 661)
        track5.car_start_angle = 53.44752724790847
        track5.walls = (((1352, 193), (1210, 244)), ((1210, 244), (908, 257)), ((908, 257), (674, 224)), ((674, 224), (433, 269)), ((433, 269), (274, 413)), ((274, 413), (209, 522)), ((209, 522), (210, 716)), ((210, 716), (302, 779)), ((302, 779), (394, 679)), ((394, 679), (426, 485)), ((426, 485), (521, 345)), ((521, 345), (665, 312)), ((665, 312), (813, 338)), ((813, 338), (938, 396)), ((938, 396), (1026, 495)), ((1026, 495), (1217, 610)), ((1217, 610), (1446, 596)), ((1446, 596), (1581, 454)), ((1581, 454), (1572, 317)), ((1572, 317), (1511, 245)), ((1511, 245), (1352, 193)), ((1295, 63), (1144, 97)), ((1144, 97), (891, 96)), ((891, 96), (648, 76)), ((648, 76), (335, 150)), ((335, 150), (139, 288)), ((139, 288), (35, 462)), ((35, 462), (32, 701)), ((32, 701), (65, 870)), ((65, 870), (186, 942)), ((186, 942), (310, 968)), ((310, 968), (410, 930)), ((410, 930), (479, 866)), ((479, 866), (550, 746)), ((550, 746), (572, 638)), ((572, 638), (630, 559)), ((630, 559), (765, 569)), ((765, 569), (847, 655)), ((847, 655), (1182, 821)), ((1182, 821), (1476, 798)), ((1476, 798), (1701, 760)), ((1701, 760), (1820, 511)), ((1820, 511), (1795, 291)), ((1795, 291), (1733, 161)), ((1733, 161), (1558, 52)), ((1558, 52), (1295, 63)))
        track5.reward_gates = (((1544, 490), (1780, 589)), ((1575, 389), (1803, 383)), ((1539, 277), (1736, 163)), ((1468, 228), (1556, 51)), ((1351, 189), (1291, 63)), ((1209, 248), (1148, 97)), ((941, 255), (918, 93)), ((675, 222), (649, 75)), ((434, 268), (335, 151)), ((311, 381), (140, 287)), ((215, 517), (32, 458)), ((210, 636), (31, 701)), ((209, 716), (67, 867)), ((302, 778), (311, 958)), ((395, 680), (548, 748)), ((415, 549), (565, 633)), ((519, 344), (630, 557)), ((665, 310), (759, 565)), ((811, 337), (842, 649)), ((1022, 494), (1018, 731)), ((1214, 610), (1182, 813)), ((1450, 596), (1565, 779)), ((1544, 490), (1780, 589)) )

        return [track1, track2, track3, track4, track5]


class TrackEditor:
    TICKS = 60

    def __init__(self, screen):
        pygame.display.set_caption('Level Editor')
        self.screen = screen
        self.exit = False
        self.direction_arrow_size = 80
        self.clock = pygame.time.Clock()

    def run(self):
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
            self.clock.tick(self.TICKS)
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
                edit_track.car_start_angle = -TrackEditor.angle_between_points(edit_track.car_start_location,
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

    @staticmethod
    def angle_between_points(point1, point2):
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return degrees(atan2(dy, dx))

    def draw_start(self, edit_track):
        if edit_track.car_start_location is not None:
            pygame.draw.circle(self.screen, RED, edit_track.car_start_location, 8)
        if edit_track.car_start_angle is not None:
            pygame.draw.line(self.screen, WHITE, edit_track.car_start_location,
                             np.add(edit_track.car_start_location,
                                    (cos(radians(edit_track.car_start_angle)) * self.direction_arrow_size,
                                     -sin(radians(edit_track.car_start_angle)) * self.direction_arrow_size)), 5)


class TrackDrawer:
    @staticmethod
    def draw_track(screen, cars, current_track):
        screen.fill(BLACK)
        TrackDrawer.draw_walls(screen, current_track)
        TrackDrawer.draw_reward_gates(screen, current_track)
        if len(cars) == 1:
            car = cars[0]
            right_gate = current_track.get_reward_gates()[car.current_reward_gate]
            pygame.draw.line(screen, BLUE, right_gate[0], right_gate[1], 4)
            wrong_gate = current_track.get_reward_gates()[car.wrong_reward_gate]
            pygame.draw.line(screen, RED, wrong_gate[0], wrong_gate[1], 4)
        TrackDrawer.draw_cars(screen, cars, current_track)
        pygame.display.update()

    @staticmethod
    def draw_walls(screen, track):
        for line in track.get_walls():
            pygame.draw.line(screen, WHITE, line[0], line[1], 3)

    @staticmethod
    def draw_reward_gates(screen, track):
        for line in track.get_reward_gates():
            pygame.draw.line(screen, GREEN, line[0], line[1], 3)

    @staticmethod
    def draw_cars(screen, cars, track):
        for car in cars:
            for line in car.vision_lines():
                pygame.draw.line(screen, WHITE, line[0], line[1])
                vision_point = Car.get_closest_vision_intersection(line, track)
                if vision_point is not None:
                    integer_point = (int(vision_point[0]), int(vision_point[1]))
                    pygame.draw.circle(screen, RED, integer_point, 8)
            for line in car.hitbox_lines():
                pygame.draw.line(screen, WHITE, line[0], line[1])
            rotated = pygame.transform.rotate(car.image, car.angle)
            rect = rotated.get_rect()
            screen.blit(rotated, car.position * car.ppu - (rect.width / 2, rect.height / 2))


class DrAIve(TrackDrawer):
    # These need to remain the same after AI has started training
    GATE_REWARD = 0.1
    FINISH_REWARD = 1
    CRASH_PUNISHMENT = -1
    FUEL_COST = 0.0001
    MAX_STALLING_TIME = 30

    OUTPUT_SHAPE = (12,)
    ALLOWED_INPUTS = 9

    def __init__(self, car_amount, screen):
        self.screen = screen
        self.car_scale, self.car_image, self.ppu = self.load_car_image()

        self.cars = []
        for i in range(car_amount):
            self.cars.append(self.build_car())

        self.tracks = Track.set_standard_tracks()
        self.track = self.tracks[1]

        self.stall_time = 0

    def init(self, track_nr):
        self.track = self.tracks[track_nr]
        self.stall_time = 0

        for i in range(len(self.cars)):
            self.cars.append(self.build_car())

    def step(self, actions, dt, draw):
        for i in range(len(self.cars)):
            self.cars[i].send_input(actions[i])
            self.cars[i].update(dt)

        results = []
        for i in range(len(self.cars)):
            results.append(self.rate_action(self.cars[i]))

        if results[0][0] != self.GATE_REWARD:
            self.stall_time += dt
            if self.stall_time > self.MAX_STALLING_TIME:
                results[0] = (self.CRASH_PUNISHMENT, True)
        else:
            self.stall_time = 0

        new_states = []
        for i in range(len(self.cars)):
            new_states.append(self.build_state(self.cars[i]))
        end_game = results[0][1]
        rewards = [i[0] for i in results]

        if draw:
            self.draw_track(self.screen, self.cars, self.track)

        return new_states, rewards, end_game

    def rate_action(self, car):  # Returns score and whether the level should be reset
        if car.hit_wall(self.track):
            return self.CRASH_PUNISHMENT, True
        elif car.hit_right_reward_gate(self.track):
            if car.current_reward_gate % len(self.track.get_reward_gates()) == 0:
                return self.FINISH_REWARD, False  # True  # CURRENTLY DOESNT STOP AFTER COMPLETING TRACK
            return self.GATE_REWARD, False
        elif car.hit_wrong_reward_gate(self.track):
            return -self.GATE_REWARD, False
        else:
            return self.FUEL_COST, False

    def build_state(self, car):
        rotated_velocity = car.velocity.rotate(car.angle)
        state = () + tuple(car.get_vision_distances(self.track)) + (rotated_velocity.x / car.top_speed, rotated_velocity.y / car.top_speed)
        # if car.hit_wall(self.track):
        #     return np.zeros(np.shape(state))
        return state

    def build_car(self):
        return Car(self.track.car_start_location[0] / self.ppu, self.track.car_start_location[1] / self.ppu,
                   self.car_scale, self.ppu, self.car_image, self.track.car_start_angle)

    def generate_tracks(self, amount):
        for i in range(amount):
            self.tracks.append(TrackEditor(self.screen).run())

    def save_tracks(self):
        track_string = ""
        for track in self.tracks:
            track_string += track.__str__() + "\n\n"
        with open("Tracks_{}.txt".format(int(time.time())), "w") as text_file:
            print(track_string, file=text_file)

    @staticmethod
    def load_car_image():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "car.png")
        car_image = pygame.image.load(image_path).convert()
        car_scale = 0.3
        new_car_image_size = (floor(car_image.get_rect().size[0] * car_scale),
                              floor(car_image.get_rect().size[1] * car_scale))
        car_image = pygame.transform.scale(car_image, new_car_image_size)
        ppu = 32 / car_scale
        return car_scale, car_image, ppu


class ManualPlayer:
    TICKS = 60

    DISPLAY_WIDTH = 1920
    DISPLAY_HEIGHT = 1080

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
        self.clock = pygame.time.Clock()
        self.tracks = []

    def run(self, make_track):
        game = DrAIve([Car(0, 0, 0, 0, None)], self.screen)
        if make_track:
            game.generate_tracks(1)
            game.init(0)
        else:
            game.init(1)
        done = False
        while not done:
            pressed = pygame.key.get_pressed()
            action = self.keys_to_choice(pressed)
            _, _, done = game.step([action], self.clock.tick(self.TICKS), True)

    @staticmethod
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
