'''
Author: Liu Gang
Date: 2020-12-21 09:33:25
LastEditTime: 2020-12-21 18:34:25
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \可行性实验程序\preprocessing.py
'''

import cv2
import argparse
import dlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

print(cv2.__version__)


class VideoPreprocess():

    def __init__(self, opt):
        self.opt = opt
        self.is_success = True

        if opt.source.split('/')[2].split('.')[0] == '1':
            if opt.distance == 'near':
                # near object of video name 1.mp4
                # self.start_frame = 1*30
                # self.end_frame = 4*30 # 4秒之前是近的

                self.start_frame = 13 * 30
                self.end_frame = 18 * 30  # 4秒之前是近的

                print('debug video 1 successful')
            elif opt.distance == 'far':
                # far object of video name 1.mp4
                self.start_frame = 6 * 30  # 6 秒之后是远的
                self.end_frame = 10 * 30  # 10秒之前是远的
        elif opt.source.split('/')[2].split('.')[0] == '2':
            if opt.distance == 'near':
                # near object of video name 2.mp4
                self.start_frame = 1 * 30
                self.end_frame = 7 * 30  # 4秒之前是近的
                print('debug video 2 successful')
            elif opt.distance == 'far':
                # far object of video name 2.mp4
                self.start_frame = 10 * 30  # 6 秒之后是远的
                self.end_frame = 18 * 30  # 10秒之前是远的

        # self.end_frame = 10
        self.predictor_path = "../dlib/shape_predictor_68_face_landmarks.dat"
        self.video_size = (200, 50)  # w h
        self.offset_pixel = 0
        self.one_time = False
        self.colors = ("b", "g", "r")
        self.channel_b = np.zeros((10000,), dtype=np.uint64)

    def video_process(self):
        # videoCapture = cv2.VideoCapture('IMG_2789.MOV')

        # dlib 相关操作
        # init
        predictor = dlib.shape_predictor(self.predictor_path)

        # 初始化dlib人脸检测器
        detector = dlib.get_frontal_face_detector()

        videoCapture = cv2.VideoCapture(self.opt.source)
        width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(videoCapture.isOpened())
        c = 0
        # self.channels = np.array()
        while self.is_success:
            self.is_success, frame = videoCapture.read()  # opencv 读取图像的格式为 H,W,C https://blog.csdn.net/sinat_28704977/article/details/89969199

            if c == 0:  # 前十秒数据
                # eye_area = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                dets = detector(frame, 0)
                shapes = []
                pt_pos = []
                for k, d in enumerate(dets):
                    print("dets{}".format(d))
                    print(
                        "Detection{}: Left: {} Top: {} Right: {} Bottom: {}".format(
                            k, d.left(), d.top(), d.right(), d.bottom()))

                    shape = predictor(frame, d)
                    # print(shape.parts().next.x)
                    for index, pt in enumerate(shape.parts()):
                        print("Part {}: {}".format(index, pt))
                        pt_pos.append((pt.x, pt.y))
                    print(pt_pos[17])

                    writer = cv2.VideoWriter(self.opt.output,
                                             cv2.VideoWriter_fourcc('I', '4',
                                                                    '2', '0'),
                                             # cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                             # cv2.VideoWriter_fourcc(*'MJPG'),
                                             30,  # fps
                                             self.video_size)  # resolution

                    # cv2.imwrite('eye_area.jpg', crop_img)
                    # cv2.imwrite('test1.jpg',frame)
                    # writer.write(crop_eye)

            c += 1
            if c >= self.start_frame and c <= self.end_frame:
                if self.one_time == False:
                    self.one_time = True
                    print('c = ', c)
                    assert c == self.start_frame, 'there are some bugs'
                dets = detector(frame, 0)
                shapes = []
                pt_pos = []
                eye_w, eye_h = 100, 50
                for k, d in enumerate(dets):
                    # print("dets{}".format(d))
                    # print("Detection{}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))

                    shape = predictor(frame, d)
                    # print(shape.parts().next.x)
                    for index, pt in enumerate(shape.parts()):
                        # print("Part {}: {}".format(index, pt))
                        pt_pos.append((pt.x, pt.y))

                    cv2.waitKey(30)
                    # crop_img = frame[pt_pos[38][1] - self.offset_pixel:pt_pos[38][1] + self.video_size[1] + self.offset_pixel, pt_pos[36][0] - self.offset_pixel:pt_pos[36][0] + self.video_size[0] + self.offset_pixel]
                    # crop_eye = cv2.resize(crop_img, self.video_size)

                    # 分别提取左右眼的区域，3通道数据
                    left_eye = frame[
                               pt_pos[37][1] - self.offset_pixel:pt_pos[37][1]
                                                                 + eye_h + self.offset_pixel,
                               pt_pos[36][0] - self.offset_pixel:pt_pos[36][
                                                                     0] + eye_w + self.offset_pixel]

                    right_eye = frame[
                                pt_pos[44][1] - self.offset_pixel:pt_pos[44][1]
                                                                  + eye_h + self.offset_pixel,
                                pt_pos[42][0] - self.offset_pixel:pt_pos[42][
                                                                      0] + eye_w + self.offset_pixel]

                    # crop_img = frame[pt_pos[38][1] - self.offset_pixel:pt_pos[38][1]
                    # + self.video_size[1] + self.offset_pixel, pt_pos[36][0] -
                    # self.offset_pixel:pt_pos[36][0] + self.video_size[0] + self.offset_pixel]
                    if opt.save_img == True:
                        # cv2.imwrite('../result/images/test/video1_near/left/N_left_eye_area_frame_{}'.format(c) + '.jpg', left_eye)
                        # cv2.imwrite('../result/images/test/video1_near/right/N_right_eye_area_frame_{}'.format(c) + '.jpg', right_eye)
                        if not os.path.exists(
                                os.path.join(opt.save_path, 'left')):
                            os.makedirs(os.path.join(opt.save_path, 'left'))

                        if not os.path.exists(
                                os.path.join(opt.save_path, 'right')):
                            os.makedirs(os.path.join(opt.save_path, 'right'))

                        if not os.path.exists(
                                os.path.join(opt.save_path, 'concat')):
                            os.makedirs(os.path.join(opt.save_path, 'concat'))

                        # 将左右眼的图像拼接在一起，为后续做直方图使用做准备
                        concat_eye = np.concatenate(
                            (left_eye[:, :, 0], right_eye[:, :, 0]),
                            axis=1)  # 提取其中一个channel绘制直方图使用
                        crop_eye = np.concatenate((left_eye, right_eye),
                                                  axis=1)
                        concat_eye_flatten = concat_eye.flatten()

                        # 防止video2中的背景中出现人脸，然后导致裁剪出来的眼睛区域尺寸不对，说明裁剪的不是目标人物的眼睛部分，故舍弃
                        # print('left_eye_shape: ', left_eye.shape)
                        if left_eye.shape[:2] == (50, 100) and right_eye.shape[:2] == (50, 100):  # 格式为 H*W*C

                            # if opt.save_img == True:  # 存储彩色的图像
                            if opt.distance == 'near':
                                cv2.imwrite(os.path.join(opt.save_path, (
                                    'left/N_left_eye_area_frame_{}.jpg'.format(
                                        c))), left_eye)
                                cv2.imwrite(os.path.join(opt.save_path, (
                                    'right/N_right_eye_area_frame_{}.jpg'.format(
                                        c))), right_eye)
                                cv2.imwrite(os.path.join(opt.save_path, (
                                    'concat/N_both_eyes_area_frame_{}.jpg'.format(
                                        c))),
                                            crop_eye)
                            else:
                                cv2.imwrite(os.path.join(opt.save_path, (
                                    'left/F_left_eye_area_frame_{}.jpg'.format(
                                        c))), left_eye)
                                cv2.imwrite(os.path.join(opt.save_path, (
                                    'right/F_right_eye_area_frame_{}.jpg'.format(
                                        c))), right_eye)
                                cv2.imwrite(os.path.join(opt.save_path, (
                                    'concat/F_both_eyes_area_frame_{}.jpg'.format(
                                        c))),
                                            crop_eye)
                            crop_eye = cv2.resize(crop_eye, self.video_size)
                            writer.write(crop_eye)
                            cv2.imshow('concat_eyes', concat_eye)

                            # concat_eye_flatten = concat_eye_flatten[np.newaxis, :]
                            # print('concat_eye.shape = ', concat_eye_flatten.shape)
                            # print('concat_eye ', concat_eye)
                            # crop_eye = cv2.resize(crop_img, self.video_size)

                            # 采集直方图数据
                            self.channel_b += concat_eye_flatten

            if c > self.end_frame:
                print('completely!')
                break

            # cv2.imshow('frame', crop_eye)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        writer.release()
        videoCapture.release()
        cv2.destroyAllWindows()

    def draw_hist(self):

        # res = pd.read_excel('../result/{}.xlsx'.format(opt.name), index_col=0)[0]
        res = pd.read_excel(os.path.join(opt.save_path, opt.name) + '.xlsx',
                            index_col=0)[0]
        plt.figure()
        print(res)
        # plt.title("Flattened Color Histogram")
        # plt.title("B_channel_near_object")
        # plt.title("video1_B_channel_near_object")
        plt.title(opt.name)
        plt.xlabel("position")
        plt.ylabel("gray_value")
        # 绘制直方图，但是这个效果不是想要的，我觉得龙教授应该是想要的柱状图
        # plt.hist(res)

        # 绘制柱状图
        plt.bar(range(len(res)), res)

        # plt.savefig('../result/{}.png'.format(opt.name))

        # plt.show()
        plt.savefig(os.path.join(opt.save_path, opt.name) + '.png', dpi=600)


if __name__ == "__main__":
    step = 1

    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov3.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='../videos/1.mp4',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str,
                        default='../result/both_eyes_concat_near.avi',
                        help='output')  # file/folder, 0 for webcam
    parser.add_argument('--name', type=str,
                        default='video1_B_channel_near_object',
                        help='inference size (pixels)')
    parser.add_argument('--distance', type=str, default='near',
                        help='测试的视频的哪一段，如果为近处，则为near 远处为far')
    parser.add_argument('--save_img', type=bool, default=True, help='是否保存每帧图像')
    parser.add_argument('--save_path', type=str,
                        default='../result/images/video1_near_2/',
                        help='保存图像的路径')
    opt = parser.parse_args()
    # print(opt)

    videoprocess = VideoPreprocess(opt)

    if step == 1:
        videoprocess.video_process()
    elif step == 2:
        videoprocess.draw_hist()  # 暂时用不到，第一步处理图像的时候就直接把图画出来了
    elif step == 3:
        # 绘制看近处和看远处的差值柱状图
        near = pd.read_excel(
            '../result/images/test/video1_near/video1_b_channel_near_object.xlsx',
            index_col=0)
        far = pd.read_excel(
            '../result/images/test/video1_far/video1_b_channel_far_object.xlsx',
            index_col=0)
        print('far', far[0])
        print('near', near[0])
        error = far[0] - near[0]

        plt.figure()

        plt.title('video1_error')
        plt.xlabel("position")
        plt.ylabel("gray_value")
        plt.bar(range(len(error)), error)
        plt.savefig('../result/images/test/video1_error.png')
        plt.show()

        plt.figure()
        sort_error = sorted(error)
        print('video1_sort_error', sort_error)
        plt.title('video1_sort_error')
        plt.xlabel("position")
        plt.ylabel("gray_value")

        plt.bar(range(len(sort_error)), sort_error)
        plt.savefig('../result/images/test/video1_sort_error.png')
        plt.show()