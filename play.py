# -*- coding:utf-8 -*-
# Created Time: 六 12/30 13:49:21 2017
# Author: Taihong Xiao <xiaotaihong@126.com>

import argparse
import glob
import os
import random
import shutil
import time

import cv2
import numpy as np


def multi_scale_search(pivot, screen, range_in=0.3, num=10):
    h_s, w_s = screen.shape[:2]
    h, w = pivot.shape[:2]

    found = None
    for scale in np.linspace(1 - range_in, 1 + range_in, num)[::-1]:
        re_sized = cv2.resize(screen, (int(w_s * scale), int(h_s * scale)))  # 重新定义截图的大小
        r = w_s / float(re_sized.shape[1])
        if re_sized.shape[0] < h or re_sized.shape[1] < w:  # 比小人或目标台面的图片大小还小 则退出
            break
        res = cv2.matchTemplate(re_sized, pivot, cv2.TM_CCOEFF_NORMED)  # 从截图中匹配出抠出的小人

        loc = np.where(res >= res.max())
        pos_h, pos_w = list(zip(*loc))[0]
        # 选取置信度（confidence score）最高的
        if found is None or res.max() > found[-1]:
            found = (pos_h, pos_w, r, res.max())

    if found is None:
        return 0, 0, 0, 0, 0
    pos_h, pos_w, r, score = found
    start_h, start_w = int(pos_h * r), int(pos_w * r)
    end_h, end_w = int((pos_h + h) * r), int((pos_w + w) * r)
    # 确定小人或目标台面（均为矩形区域）的起始位置
    return [start_h, start_w, end_h, end_w, score]


class WechatAutoJump(object):
    def __init__(self, phone, sensitivity, debug, resource_dir):
        self.phone = phone
        self.sensitivity = sensitivity
        self.debug = debug
        self.resource_dir = resource_dir
        self.bb_size = [300, 300]
        self.step = 0
        self.player = None
        self.jump_file = None
        self.resolution = None
        self.state = None
        self.player_pos = None
        self.target_pos = None

        self.load_resource()

        if self.phone == 'IOS':  # for IOS, WebDriverAgent
            import wda
            self.client = wda.Client('http://localhost:8100')
            self.sess = self.client.session()
        if self.debug:  # 创建debug目录
            if not os.path.exists(self.debug):
                os.mkdir(self.debug)

    def load_resource(self):
        self.player = cv2.imread(os.path.join(self.resource_dir, 'player.png'), 0)  # 抠出的跳一跳-小人
        circle_file = glob.glob(os.path.join(self.resource_dir, 'circle/*.png'))  # 抠出的圆形台面
        table_file = glob.glob(os.path.join(self.resource_dir, 'table/*.png'))  # 抠出的方形台面
        self.jump_file = [cv2.imread(name, 0) for name in circle_file + table_file]

    def get_current_state(self):
        if self.phone == 'Android':
            os.system('adb shell screencap -p /sdcard/1.png')  # 截取图片
            os.system('adb pull /sdcard/1.png state.png')  # 获取截取的图片-重命名为state.png
        elif self.phone == 'IOS':
            self.client.screenshot('state.png')

        if self.debug:  # debug是个目录
            shutil.copyfile('state.png', os.path.join(self.debug, 'state_{:03d}.png'.format(self.step)))

        state = cv2.imread('state.png')  # 读取截取的图片
        self.resolution = state.shape[:2]  # 截图分辨率/大小 height 与 width: state.shape[0] 与 state.shape[1]
        scale = state.shape[1] / 720.
        state = cv2.resize(state, (720, int(state.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)
        # 始终将截图重新定义为大小为1280*720
        if state.shape[0] > 1280:  # height
            s = state.shape[0] - 1280
            state = state[s:, :, :]
        elif state.shape[0] < 1280:
            s = 1280 - state.shape[0]
            state = np.concatenate((255 * np.ones((s, 720, 3), dtype=np.uint8), state), 0)
        return state

    # 注意到小人在屏幕的不同位置，大小略有不同，设计了多尺度的搜索，用不同大小的进行匹配，最后选取置信度（confidence score）最高的
    def get_player_position(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        pos = multi_scale_search(self.player, state, 0.3, 10)  # 多尺度搜索：小人、截图、变化范围、尺度数
        h, w = int((pos[0] + 13 * pos[2]) / 14.), (pos[1] + pos[3]) // 2
        return np.array([h, w])

    # 1、注意到目标位置始终在小人的位置的上面,找到小人位置之后把小人位置以下的部分都舍弃掉,减少搜索空间
    # 2、小人和目标台面基本上是关于屏幕中心对称的位置,假设屏幕分辨率是(1280，720), 小人底部的位置是(h1, w1),
    # 那么关于中心对称点的位置就是(1280 - h1, 720 - w1),以这个点为中心的一个边长300的正方形内，再去多尺度搜索目标位置
    def get_target_position(self, state, player_pos):
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        sym_center = np.array([1280, 720]) - player_pos  # list 减法
        sym_tl = np.maximum([0, 0], sym_center + np.array([-self.bb_size[0] // 2, -self.bb_size[1] // 2]))
        sym_br = np.array(
            [min(sym_center[0] + self.bb_size[0] // 2, player_pos[0]), min(sym_center[0] + self.bb_size[1] // 2, 720)])
        # 边长300的正方形区域
        state_cut = state[sym_tl[0]:sym_br[0], sym_tl[1]:sym_br[1]]
        target_pos = None
        for target in self.jump_file:
            pos = multi_scale_search(target, state_cut, 0.4, 15)
            # 选取置信度（confidence score）最高的
            if target_pos is None or pos[-1] > target_pos[-1]:
                target_pos = pos
        return np.array([(target_pos[0] + target_pos[2]) // 2, (target_pos[1] + target_pos[3]) // 2]) + sym_tl

    # 小人上一次如果跳到台面中心，那么下一次目标台面的中心会有一个白点，白点的RGB值是（245, 245, 245）
    # 直接去搜索这个白点，注意到白点是一个连通区域，像素值为（245，245，245）的像素个数稳定在280-310之间，
    # 所以我们可以利用这个去直接找到目标的位置
    @staticmethod
    def get_target_position_fast(state, player_pos):
        r_x, r_y = None, None
        state_cut = state[:player_pos[0], :, :]
        m1 = (state_cut[:, :, 0] == 245)
        m2 = (state_cut[:, :, 1] == 245)
        m3 = (state_cut[:, :, 2] == 245)
        m = np.uint8(np.float32(m1 * m2 * m3) * 255)
        b1, b2 = cv2.connectedComponents(m)
        for i in range(1, np.max(b2) + 1):
            x, y = np.where(b2 == i)
            # print('fast', len(x))
            if 280 < len(x) < 310:  # 比较大小的简化写法
                r_x, r_y = x, y
        h, w = int(r_x.mean()), int(r_y.mean())
        return np.array([h, w])

    def jump(self, player_pos, target_pos):
        distance = np.linalg.norm(player_pos - target_pos)
        # 两点距离乘以一个系数=按压时间
        press_time = distance * self.sensitivity
        press_time = int(press_time)
        if self.phone == 'Android':
            # 按压点在一定范围内随机
            press_h, press_w = random.randint(300, 800), random.randint(200, 800)
            cmd = 'adb shell input swipe {} {} {} {} {}'.format(press_w, press_h, press_w, press_h, press_time)
            print(cmd)
            os.system(cmd)
        elif self.phone == 'IOS':
            self.sess.tap_hold(random.randint(300, 800), random.randint(200, 800), press_time / 1000.)

    def debugging(self):
        current_state = self.state.copy()
        # 标出小人位置 绿点
        cv2.circle(current_state, (self.player_pos[1], self.player_pos[0]), 5, (0, 255, 0), -1)
        # 标出目标位置 红点
        cv2.circle(current_state, (self.target_pos[1], self.target_pos[0]), 5, (0, 0, 255), -1)
        # 保存在路径下
        cv2.imwrite(os.path.join(self.debug, 'state_{:03d}_res_h_{}_w_{}.png'.format(self.step, self.target_pos[0],
                                                                                     self.target_pos[1])),
                    current_state)

    def play(self):
        self.state = self.get_current_state()
        print(self.state.shape)  # (1280, 720, 3)
        # 小人位置坐标
        self.player_pos = self.get_player_position(self.state)
        # 目标台面位置坐标
        if self.phone == 'IOS':
            self.target_pos = self.get_target_position(self.state, self.player_pos)
            print('multi-scale-search, step: %04d' % self.step)
        else:
            try:
                self.target_pos = self.get_target_position_fast(self.state, self.player_pos)
                print('fast-search, step: %04d' % self.step)
            except Exception as ex:
                print("play exception: " + str(ex))
                self.target_pos = self.get_target_position(self.state, self.player_pos)
                print('multi-scale-search, step: %04d' % self.step)
        if self.debug:
            self.debugging()
        # 跳跃
        self.jump(self.player_pos, self.target_pos)
        self.step += 1
        # 等待时间 1~2秒随机
        ts = random.uniform(1, 2)
        time.sleep(ts)

    def run(self):
        try:
            while True:
                self.play()
        except KeyboardInterrupt:
            pass
        # self.play()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--phone', default='Android', choices=['Android', 'IOS'], type=str, help='mobile phone OS')
    parser.add_argument('--sensitivity', default=2.045, type=float, help='constant for press time')
    parser.add_argument('--resource', default='resource', type=str, help='resource dir')
    parser.add_argument('--debug', default='debug_pics', type=str,
                        help='debug mode, specify a directory for storing log files.')
    args = parser.parse_args()
    # print(args)

    AI = WechatAutoJump(args.phone, args.sensitivity, args.debug, args.resource)
    AI.run()
