import os
from multiprocessing import Pool

import numpy as np
import cv2 as cv

PITCH_HALF_WIDTH = 34
PITCH_HALF_LENGTH = 52.5
N_DATA = 0

class Player:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.unum = 0
        self.side = 0


class Ball:
    def __init__(self):
        self.x = 0
        self.y = 0


def create_headrs():
    header_lst = 'cycle,ball_pos_x,ball_pos_y,ball_pos_r,ball_pos_t,p_l_1_unum,p_l_1_pos_x,p_l_1_pos_y,p_l_2_unum,p_l_2_pos_x,p_l_2_pos_y,p_l_3_unum,p_l_3_pos_x,p_l_3_pos_y,p_l_4_unum,p_l_4_pos_x,p_l_4_pos_y,p_l_5_unum,p_l_5_pos_x,p_l_5_pos_y,p_l_6_unum,p_l_6_pos_x,p_l_6_pos_y,p_l_7_unum,p_l_7_pos_x,p_l_7_pos_y,p_l_8_unum,p_l_8_pos_x,p_l_8_pos_y,p_l_9_unum,p_l_9_pos_x,p_l_9_pos_y,p_l_10_unum,p_l_10_pos_x,p_l_10_pos_y,p_l_11_unum,p_l_11_pos_x,p_l_11_pos_y,p_r_1_unum,p_r_1_pos_x,p_r_1_pos_y,p_r_2_unum,p_r_2_pos_x,p_r_2_pos_y,p_r_3_unum,p_r_3_pos_x,p_r_3_pos_y,p_r_4_unum,p_r_4_pos_x,p_r_4_pos_y,p_r_5_unum,p_r_5_pos_x,p_r_5_pos_y,p_r_6_unum,p_r_6_pos_x,p_r_6_pos_y,p_r_7_unum,p_r_7_pos_x,p_r_7_pos_y,p_r_8_unum,p_r_8_pos_x,p_r_8_pos_y,p_r_9_unum,p_r_9_pos_x,p_r_9_pos_y,p_r_10_unum,p_r_10_pos_x,p_r_10_pos_y,p_r_11_unum,p_r_11_pos_x,p_r_11_pos_y,out_target_x,out_target_y,out_unum'
    header_lst = header_lst.split(',')
    header = {}
    for i, h in enumerate(header_lst):
        header[h] = i

    return header


def draw_player(img, p: Player, color, r):
    margin = r
    h = img.shape[0] - 2 * margin
    w = img.shape[1] - 2 * margin
    center_index_h = round((p.y + PITCH_HALF_WIDTH) / (PITCH_HALF_WIDTH * 2) * h) + margin
    center_index_w = round((p.x + PITCH_HALF_LENGTH) / (PITCH_HALF_LENGTH * 2) * w) + margin

    area = img[center_index_h - r: center_index_h + r, center_index_w - r: center_index_w + r]

    r2 = r * r

    for i in range(2 * r):
        for j in range(2 * r):
            if (i - r) ** 2 + (j - r) ** 2 < r2:
                area[i][j] = color


def make_image(ball: Ball, players: list[Player]):
    w, h = 256, 128
    player_r = 10
    ball_r = 5
    left_player_color = np.array([255., 0., 0.])
    right_player_color = np.array([0., 255., 0.])
    ball_color = np.array([0., 0., 255.])

    img = np.zeros((h, w, 3))

    for p in players:
        draw_player(img, p, left_player_color if p.side == 0 else right_player_color, player_r)

    draw_player(img, ball, ball_color, ball_r)
    return img


def draw_field(inps):
    data = inps[0]
    n = inps[1]
    print(f'{n}\t{N_DATA}')
    ball_data = data[1:5]
    players_data = data[5:-3]
    out_data = data[-3:]
    print(data)

    ball = Ball()
    ball.x = ball_data[0]
    ball.y = ball_data[1]

    players = [Player() for _ in range(22)]

    for i, p in enumerate(players):
        p.unum = int(players_data[i * 3])
        p.x = players_data[i * 3 + 1]
        p.y = players_data[i * 3 + 2]
        p.side = 0 if i <= 10 else 1
    img = make_image(ball, players)
    return img, out_data


dir_data = '/data1/aref/2d/data/'
files_list = os.listdir(dir_data)
n = 0
inps = []
for file in files_list:
    if not file.endswith('.csv'):
        continue
    lines = open(f'{dir_data}{file}', 'r').readlines()
    for line in lines[1:]:
        data = np.fromstring(line, dtype=float, sep=',')
        inps.append((data, n))
        n += 1
    if n > 3000:
        break

N_DATA = n

pool = Pool(1)
res = pool.map(draw_field, inps)

x = []
y = []
n_r = len(res)
for i, r in enumerate(res):
    print(i, n_r)
    x.append(r[0])
    y.append(r[1])

x = np.array(x)
y = np.array(y)

np.save('x.npy', x)
np.save('y.npy', y)

