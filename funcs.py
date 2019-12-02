from collections import namedtuple
import os
import glob
import numpy as np
from navpy import lla2ned
import torch
import func_math as fm
import datetime
import pickle
from collections import OrderedDict
import torch.nn as nn
import sys

pickle_extension = ".p"


def read_KITTI_data(data_directory):
    """
    Read the data from the KITTI dataset
    """
    print("Start reading KITTI dataset from", data_directory)

    # COUNT AVAILABLE DATASET
    data_set_counter = 0
    date_dirs = os.listdir(data_directory)
    max_data_length = 0
    for n_iter, date_dir in enumerate(date_dirs):
        # get access to each sequence
        path1 = os.path.join(data_directory, date_dir)
        if not os.path.isdir(path1):
            continue
        date_dirs2 = os.listdir(path1)
        for date_dir2 in date_dirs2:
            path2 = os.path.join(path1, date_dir2)
            if not os.path.isdir(path2):
                continue
            oxts_files = sorted(glob.glob(os.path.join(path2, 'oxts', 'data', '*.txt')))
            oxts = load_oxts_packets_and_poses(oxts_files)
            data_set_counter = data_set_counter + 1
            if max_data_length < len(oxts):
                max_data_length = len(oxts)

    # the size of the dataset
    print("Number of available dataset: ", data_set_counter)
    print("Max data length: ", max_data_length)
    data_lengths = np.zeros(data_set_counter)
    t = np.zeros((data_set_counter, max_data_length))
    lat_oxts = np.zeros((data_set_counter, max_data_length))
    lon_oxts = np.zeros((data_set_counter, max_data_length))
    alt_oxts = np.zeros((data_set_counter, max_data_length))
    roll_oxts = np.zeros((data_set_counter, max_data_length))
    pitch_oxts = np.zeros((data_set_counter, max_data_length))
    yaw_oxts = np.zeros((data_set_counter, max_data_length))
    roll_gt = np.zeros((data_set_counter, max_data_length))
    pitch_gt = np.zeros((data_set_counter, max_data_length))
    yaw_gt = np.zeros((data_set_counter, max_data_length))
    acc = np.zeros((data_set_counter, max_data_length, 3))
    acc_bis = np.zeros((data_set_counter, max_data_length, 3))
    gyro = np.zeros((data_set_counter, max_data_length, 3))
    gyro_bis = np.zeros((data_set_counter, max_data_length, 3))
    p_gt = np.zeros((data_set_counter, max_data_length, 3))
    v_gt = np.zeros((data_set_counter, max_data_length, 3))
    v_rob_gt = np.zeros((data_set_counter, max_data_length, 3))
    ang_gt = np.zeros((data_set_counter, max_data_length, 3))

    # READ THE REAL DATASET
    dsc = 0
    for n_iter, date_dir in enumerate(date_dirs):
        # get access to each sequence
        path1 = os.path.join(data_directory, date_dir)
        if not os.path.isdir(path1):
            continue
        date_dirs2 = os.listdir(path1)
        for date_dir2 in date_dirs2:
            path2 = os.path.join(path1, date_dir2)
            if not os.path.isdir(path2):
                continue
            # read data
            oxts_files = sorted(glob.glob(os.path.join(path2, 'oxts', 'data', '*.txt')))
            oxts = load_oxts_packets_and_poses(oxts_files)

            """ Note on difference between ground truth and oxts solution:
                - orientation is the same
                - north and east axis are inverted
                - position are closed to but different
                => oxts solution is not loaded
            """

            print("Loaded KITTI data : " + date_dir2)
            data_lengths[dsc] = len(oxts)
            t_tmp = load_timestamps(path2)
            k_max = len(oxts)
            for k in range(k_max):
                oxts_k = oxts[k]
                # print(dsc)
                # print(k)
                t[dsc, k] = 3600 * t_tmp[k].hour + 60 * t_tmp[k].minute + t_tmp[k].second + t_tmp[k].microsecond / 1e6
                lat_oxts[dsc, k] = oxts_k[0].lat
                lon_oxts[dsc, k] = oxts_k[0].lon
                alt_oxts[dsc, k] = oxts_k[0].alt
                acc[dsc, k, 0] = oxts_k[0].af
                acc[dsc, k, 1] = oxts_k[0].al
                acc[dsc, k, 2] = oxts_k[0].au
                acc_bis[dsc, k, 0] = oxts_k[0].ax
                acc_bis[dsc, k, 1] = oxts_k[0].ay
                acc_bis[dsc, k, 2] = oxts_k[0].az
                gyro[dsc, k, 0] = oxts_k[0].wf
                gyro[dsc, k, 1] = oxts_k[0].wl
                gyro[dsc, k, 2] = oxts_k[0].wu
                gyro_bis[dsc, k, 0] = oxts_k[0].wx
                gyro_bis[dsc, k, 1] = oxts_k[0].wy
                gyro_bis[dsc, k, 2] = oxts_k[0].wz
                roll_oxts[dsc, k] = oxts_k[0].roll
                pitch_oxts[dsc, k] = oxts_k[0].pitch
                yaw_oxts[dsc, k] = oxts_k[0].yaw
                v_gt[dsc, k, 0] = oxts_k[0].ve
                v_gt[dsc, k, 1] = oxts_k[0].vn
                v_gt[dsc, k, 2] = oxts_k[0].vu
                v_rob_gt[dsc, k, 0] = oxts_k[0].vf
                v_rob_gt[dsc, k, 1] = oxts_k[0].vl
                v_rob_gt[dsc, k, 2] = oxts_k[0].vu
                p_gt[dsc, k] = oxts_k[1][:3, 3]
                ang_gt[dsc, k, 0] = oxts_k[0].roll
                ang_gt[dsc, k, 1] = oxts_k[0].pitch
                ang_gt[dsc, k, 2] = oxts_k[0].yaw

        dsc = dsc + 1
    # convert from numpy
    t = torch.from_numpy(t)
    p_gt = torch.from_numpy(p_gt)
    v_gt = torch.from_numpy(v_gt)
    ang_gt = torch.from_numpy(ang_gt)
    gyro_bis = torch.from_numpy(gyro_bis)
    acc_bis = torch.from_numpy(acc_bis)

    # convert to float
    t = t.float()
    p_gt = p_gt.float()
    v_gt = v_gt.float()
    ang_gt = ang_gt.float()
    gyro_bis = gyro_bis.float()
    acc_bis = acc_bis.float()

    if (len(data_lengths) < 3):
        print("Error: We need at least 2 training data and a test dataset")
        sys.exit()
    else:
        t_train = t[0:-1, :]
        data_lengths_train = data_lengths[0:-1]
        p_gt_train = p_gt[0:-1, :]
        v_gt_train = v_gt[0:-1, :]
        ang_gt_train = ang_gt[0:-1, :]
        gyro_bis_train = gyro_bis[0:-1, :]
        acc_bis_train = acc_bis[0:-1, :]

        t_test = t[-1, :]
        data_lengths_test = data_lengths[-1]
        p_gt_test = p_gt[-1, :]
        v_gt_test = v_gt[-1, :]
        ang_gt_test = ang_gt[-1, :]
        gyro_bis_test = gyro_bis[-1, :]
        acc_bis_test = acc_bis[-1, :]

        print("The number of train dataset: ", data_lengths_train.shape[0])
    return t_train, data_lengths_train, p_gt_train, v_gt_train, ang_gt_train, gyro_bis_train, acc_bis_train, t_test, data_lengths_test, p_gt_test, v_gt_test, ang_gt_test, gyro_bis_test, acc_bis_test


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out


def get_inputs(i, j, gyro_bis, acc_bis, batch_size):
    if gyro_bis.ndim > 2:  # if this is a training dataset
        # create single output
        w = gyro_bis[i][j]
        a = acc_bis[i][j]
        u = torch.cat((w, a))  # This is single measurement
        u = u.view(1, 1, 6)
        us = u
        # concatenate more
        for k in range(batch_size - 1):
            # create single output
            w = gyro_bis[i][j + k + 1]
            a = acc_bis[i][j + k + 1]
            u = torch.cat((w, a))  # This is single measurement
            u = u.view(1, 1, 6)
            us = torch.cat((us, u), 0)
    else:  # this is a test dataset
        # create single output
        w = gyro_bis[j]
        a = acc_bis[j]
        u = torch.cat((w, a))  # This is single measurement
        u = u.view(1, 1, 6)
        us = u
        # concatenate more
        for k in range(batch_size - 1):
            # create single output
            w = gyro_bis[j + k + 1]
            a = acc_bis[j + k + 1]
            u = torch.cat((w, a))  # This is single measurement
            u = u.view(1, 1, 6)
            us = torch.cat((us, u), 0)
    return us


def get_labels(i, j, p_gt, v_gt, ang_gt, gyro_bis, acc_bis, batch_size):
    if gyro_bis.ndim > 2:  # if this is a training dataset
        # create label
        labels = torch.tensor([0.0])
        if v_gt[i][j].norm() < 0.1:
            labels = torch.tensor([1.0])
        # concatenate more
        for k in range(batch_size - 1):
            label_add = torch.tensor([0.0])
            if v_gt[i][j + k + 1].norm() < 0.1:
                label_add = torch.tensor([1.0])
            labels = torch.cat((labels, label_add), 0)
    else:  # this is a test dataset
        labels = torch.tensor([0.0])
        if v_gt[j].norm() < 0.1:
            labels = torch.tensor([1.0])
        # concatenate more
        for k in range(batch_size - 1):
            label_add = torch.tensor([0.0])
            if v_gt[j + k + 1].norm() < 0.1:
                label_add = torch.tensor([1.0])
            labels = torch.cat((labels, label_add), 0)

    return labels


def load_oxts_packets_and_poses(oxts_files):
    """Generator to read OXTS ground truth data.
       Poses are given in an East-North-Up coordinate system
       whose origin is the first GPS position.
    """
    # Scale for Mercator projection (from first lat value)
    scale = None
    # Origin of the global coordinate system (first GPS position)
    origin = None

    oxts = []

    for filename in oxts_files:
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.split()
                # Last five entries are flags and counts
                line[:-5] = [float(x) for x in line[:-5]]
                line[-5:] = [int(float(x)) for x in line[-5:]]

                # Rap the name format
                packet = KITTIDataset.OxtsPacket(*line)

                if scale is None:
                    scale = np.cos(packet.lat * np.pi / 180.)

                R, t = fm.pose_from_oxts_packet(packet, scale)

                if origin is None:
                    origin = t
                IMU_poses_global = fm.R_t_2_se3(R, t - origin)

                oxts.append(KITTIDataset.OxtsData(packet, IMU_poses_global))
    return oxts

def load_timestamps(data_path):
    """Load timestamps from file."""
    timestamp_file = os.path.join(data_path, 'oxts', 'timestamps.txt')

    # Read and parse the timestamps
    timestamps = []
    with open(timestamp_file, 'r') as f:
        for line in f.readlines():
            # NB: datetime only supports microseconds, but KITTI timestamps
            # give nanoseconds, so need to truncate last 4 characters to
            # get rid of \n (counts as 1) and extra 3 digits
            t = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
            timestamps.append(t)
    return timestamps


class KITTIDataset():
    OxtsPacket = namedtuple('OxtsPacket',
                            'lat, lon, alt, ' + 'roll, pitch, yaw, ' + 'vn, ve, vf, vl, vu, '
                                                                       '' + 'ax, ay, az, af, al, '
                                                                            'au, ' + 'wx, wy, wz, '
                                                                                     'wf, wl, wu, '
                                                                                     '' +
                            'pos_accuracy, vel_accuracy, ' + 'navstat, numsats, ' + 'posmode, '
                                                                                    'velmode, '
                                                                                    'orimode')

    # Bundle into an easy-to-access structure
    OxtsData = namedtuple('OxtsData', 'packet, IMU_poses_global')

    def __init__(self):
        self.datasets_validatation_filter = OrderedDict()
        """Validation dataset with index for starting/ending"""
        self.datasets_train_filter = OrderedDict()
        """Validation dataset with index for starting/ending"""
        super(KITTIDataset, self).__init__()
