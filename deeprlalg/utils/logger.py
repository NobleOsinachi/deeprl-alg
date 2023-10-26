import os
from tensorboardX import SummaryWriter
import numpy as np
import torchvision

import os.path as osp, shutil, time, atexit, os, subprocess

from deeprlalg.user_config import DEFAULT_DATA_DIR
from deeprlalg.utils.utils import colorize
import json


class Logger:
    def __init__(self, log_dir, n_logged_samples=10, summary_writer=None):
        self._log_dir = log_dir

        if not(os.path.exists(self._log_dir)):
            os.makedirs(self._log_dir)
            f = open(osp.join(self._log_dir, "log.txt"), 'w')
            atexit.register(f.close)
            print('########################')
            print(colorize("Logging data to %s"%f.name, 'green', bold=True))
            print('########################')

        self._n_logged_samples = n_logged_samples
        self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)

    def log_scalar(self, scalar, name, step_):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)

    def log_scalars(self, scalar_dict, group_name, step, phase):
        """Will log all scalars in the same plot."""
        self._summ_writer.add_scalars('{}_{}'.format(group_name, phase), scalar_dict, step)

    def log_image(self, image, name, step):
        assert(len(image.shape) == 3)  # [C, H, W]
        self._summ_writer.add_image('{}'.format(name), image, step)

    def log_video(self, video_frames, name, step, fps=10):
        assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
        # torchvision.io.write_video(self._log_dir,video_frames, fps)
        # print(name, video_frames)
        self._summ_writer.add_video('{}'.format(name), video_frames, step, fps=fps)

    def log_paths_as_videos(self, paths, step, max_videos_to_save=2, fps=10, video_title='video'):


        # reshape the rollouts
        videos = [np.transpose(p['image_obs'], [0, 3, 1, 2]) for p in paths]

        # max rollout length
        max_videos_to_save = np.min([max_videos_to_save, len(videos)])
        max_length = videos[0].shape[0]
        for i in range(max_videos_to_save):
            if videos[i].shape[0]>max_length:
                max_length = videos[i].shape[0]

        # pad rollouts to all be same length
        for i in range(max_videos_to_save):
            if videos[i].shape[0]<max_length:
                padding = np.tile([videos[i][-1]], (max_length-videos[i].shape[0],1,1,1))
                videos[i] = np.concatenate([videos[i], padding], 0)

        # log videos to tensorboard event file
        videos = np.stack(videos[:max_videos_to_save], 0)

        self.log_video(videos, video_title, step, fps=fps)

    def log_figures(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        assert figure.shape[0] > 0, "Figure logging requires input shape [batch x figures]!"
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    def log_figure(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    def log_graph(self, array, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        im = plot_graph(array)
        self._summ_writer.add_image('{}_{}'.format(name, phase), im, step)

    def dump_scalars(self, log_path=None):
        log_path = os.path.join(self._log_dir, "scalar_data.json") if log_path is None else log_path
        self._summ_writer.export_scalars_to_json(log_path)

    def dump_tabular(self, itr, log, print_data=True):
        """
        Write all of the diagnostics from the current iteration
        """
        # vals = []
        if print_data:
            key_lens = [len(key) for key in log.keys()]
            max_key_len = max(15,max(key_lens))
            keystr = '%'+'%d'%max_key_len
            fmt = "| " + keystr + "s | %15s |"
            n_slashes = 22 + max_key_len
            print("-"*n_slashes)
            for key in log.keys():
                val = log.get(key, "")
                if hasattr(val, "__float__"): valstr = "%8.3g"%val
                else: valstr = val
                print(fmt%(key, valstr))
                # vals.append(val)
            print("-"*n_slashes)


        fname = self._log_dir +'/log.txt'
        if itr == 0:
            with open(fname, 'w') as f:
                f.write('\t'.join(log.keys()))
                f.write('\n')

        else:
            with open(fname, 'a') as f:
                vals = list(log.values())
                f.write('\t'.join(map(str,vals)))
                f.write('\n')

    def save_params(self, params):
        with open(self._log_dir+"/params.json", "w") as outfile:
            json.dump(params, outfile, indent=1)

        print(colorize('Saving parameters:\n', color='cyan', bold=True))
        print(params)


    def flush(self):
        self._summ_writer.flush()
