import copy

import numpy as np

from .box import Box3D
from .kalman_filter import KF
from .matching import data_association


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


class AB3DMOT():
    def __init__(self, category, ID_init=0):

        # counter
        self.trackers = []
        self.frame_count = 0
        self.ID_count = [ID_init]
        self.id_now_output = []

        # config
        self.category = category

        # 对不同的类别设置不同的关联方法
        if category == 'Car':
            algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.2, 3, 2
        elif category == 'Pedestrian':
            algm, metric, thres, min_hits, max_age = 'greedy', 'dist_3d', 2, 3, 4
        elif category == 'Cyclist':
            algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 2, 3, 4
        else:
            assert False, 'error'

        # add negative due to it is the cost
        if metric in ['dist_3d', 'dist_2d', 'm_dis']:
            thres *= -1
        self.algm, self.metric, self.thres, self.max_age, self.min_hits = \
            algm, metric, thres, max_age, min_hits

        # 设置相似度矩阵的最大值和最小值
        if self.metric in ['dist_3d', 'dist_2d', 'm_dis']:
            self.max_sim, self.min_sim = 0.0, -100.
        elif self.metric in ['iou_2d', 'iou_3d']:
            self.max_sim, self.min_sim = 1.0, 0.0
        elif self.metric in ['giou_2d', 'giou_3d']:
            self.max_sim, self.min_sim = 1.0, -1.0

    def process_dets(self, dets):
        # 将其转换为类
        dets_new = []
        for det in dets:
            det_tmp = Box3D.array2bbox_raw(det)
            dets_new.append(det_tmp)

        return dets_new

    def within_range(self, theta):
        # make sure the orientation is within a proper range

        if theta >= np.pi: theta -= np.pi * 2  # make the theta still in the range
        if theta < -np.pi: theta += np.pi * 2

        return theta

    def orientation_correction(self, theta_pre, theta_obs):
        # update orientation in propagated tracks and detected boxes so that they are within 90 degree

        # make the theta still in the range
        theta_pre = self.within_range(theta_pre)
        theta_obs = self.within_range(theta_obs)

        # if the angle of two theta is not acute angle, then make it acute
        if np.pi / 2.0 < abs(theta_obs - theta_pre) < np.pi * 3 / 2.0:
            theta_pre += np.pi
            theta_pre = self.within_range(theta_pre)

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(theta_obs - theta_pre) >= np.pi * 3 / 2.0:
            if theta_obs > 0:
                theta_pre += np.pi * 2
            else:
                theta_pre -= np.pi * 2

        return theta_pre, theta_obs

    def prediction(self):
        """
        基于当前帧的状态，预测下一帧的状态
        """
        trackers = []
        for t in range(len(self.trackers)):
            # propagate locations
            kf_tmp = self.trackers[t]
            kf_tmp.kf.x[3] = self.within_range(kf_tmp.kf.x[3])

            # update statistics
            kf_tmp.time_since_update += 1
            trk_tmp = kf_tmp.kf.x.reshape((-1))[:7]
            trackers.append(Box3D.array2bbox(trk_tmp))

        return trackers

    def update(self, matched, unmatched_trackers, detections):
        """
        数据融合：预测值+观测值进行融合
        """
        detections = copy.copy(detections)
        for t, tracker in enumerate(self.trackers):
            if t not in unmatched_trackers:
                d = matched[np.where(matched[:, 1] == t)[0], 0]  # a list of index
                assert len(d) == 1, 'error'

                # reset because just updated
                tracker.time_since_update = 0
                tracker.hits += 1

                # update orientation in propagated tracks and detected boxes so that they are within 90 degree
                bbox3d = Box3D.bbox2array(detections[d[0]])
                tracker.kf.x[3], bbox3d[3] = self.orientation_correction(tracker.kf.x[3], bbox3d[3])
                tracker.kf.update(bbox3d)

                tracker.kf.x[3] = self.within_range(tracker.kf.x[3])

    def birth(self, detections, unmatched_detections_indices):
        """
        生存周期管理模块
        :param detections:
        :param info:
        :param unmatched_detections_indices:
        :return:
        """

        for i in unmatched_detections_indices:
            trk = KF(Box3D.bbox2array(detections[i]), self.ID_count[0])
            self.trackers.append(trk)
            self.ID_count[0] += 1

    def output(self):
        """
        输出有多次跟踪数据的tracker
        删除很久没更新的tracker
        :return:
        """

        num_trks = len(self.trackers)
        results = []
        for trk in reversed(self.trackers):
            # change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
            d = Box3D.array2bbox(trk.kf.x[:7].reshape((7,)))  # bbox location self
            d = Box3D.bbox2array_raw(d)

            if ((trk.time_since_update < self.max_age) and (
                    trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
                results.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            num_trks -= 1

            # 移除很久没更新的tracker
            if trk.time_since_update >= self.max_age:
                self.trackers.pop(num_trks)

        return results

    def track(self, detections):
        """ 输入每一帧的检测结果，输出每一帧的跟踪结果
        Note: 即便是空帧，也需要提供一个空的detections，否则会出错
        :param detections:  (N, 7) [x, y, z, l, w, h, yaw] （激光雷达系）
        :return: (N, 8) [x, y, z, l, w, h, yaw, id]（激光雷达系）
        """

        self.frame_count += 1

        # 类型转换 numpy -> object
        detections = self.process_dets(detections)

        # 步骤一：状态预测
        trackers = self.prediction()

        # 步骤二：运动补偿
        # TODO

        # 步骤三：数据关联
        trk_innovation_matrix = None
        if self.metric == 'm_dis':
            trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in self.trackers]
        matched, unmatched_dets, unmatched_trks, cost, affi = \
            data_association(detections, trackers, self.metric,
                             self.thres, self.algm, trk_innovation_matrix)

        # 步骤四：数据更新：使用关联的数据更新tracker
        self.update(matched, unmatched_trks, detections)

        # 对没有匹配的检测结果，初始化一个tracker
        self.birth(detections, unmatched_dets)

        # 输出tracker
        results = self.output()
        if len(results) > 0:
            self.id_now_output = results[0][:, 7].tolist()
            return results, affi
        else:
            return [], affi
