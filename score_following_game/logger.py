from __future__ import annotations

import getpass
import os
import sys
import time
import torch

import numpy as np

from score_following_game.agents import models
from torch.utils.tensorboard import SummaryWriter
from typing import Optional


RUNNING_AVG_STEPS = 11


class Logger:

    def __init__(self, log_writer: Optional[SummaryWriter], log_gradients: bool = False,
                 log_interval: int = 100, grad_log_interval: int = 100, dump_dir: Optional[str] = None,
                 dump_interval: int = 100000):

        self.log_writer = log_writer
        self.log_gradients = log_gradients
        self.log_interval = log_interval
        self.grad_log_interval = grad_log_interval
        self.dump_dir = dump_dir
        self.dump_interval = dump_interval
        self.now = self.after = time.time()
        self.step_times = np.ones(RUNNING_AVG_STEPS, dtype=np.float32)

    def log(self, model: models.Model, log_dict: dict, stage: str, update_cnt: int) -> None:

        if update_cnt % self.log_interval == 0 and update_cnt > 0:

            print('-' * 32)
            print('| {:<15} {: 12d} |'.format('update', update_cnt))
            print('| {:<15} {: 12.1f} |'.format('duration(s)', time.time() - self.now))
            for log_key in log_dict:

                log_var = log_dict[log_key].cpu().item() if isinstance(log_dict[log_key], torch.Tensor)\
                    else log_dict[log_key]

                if type(log_dict[log_key]) == int:
                    print('| {:<15} {: 12d} |'.format(log_key, log_var))
                else:
                    print('| {:<15} {: 12.5f} |'.format(log_key, log_var))

                if self.log_writer is not None:
                    self.log_writer.add_scalar(f'{stage}/{log_key}', log_var, int(update_cnt / self.log_interval))

            print('-' * 32)
            self.now = time.time()

        if self.log_writer is not None and self.log_gradients and update_cnt % self.grad_log_interval == 0:

            for tag, value in model.net.named_parameters():
                if value.grad is not None:
                    self.log_writer.add_scalar(f'gradient_norms/{tag}', value.grad.data.cpu().norm(2).item(),
                                               int(update_cnt / self.log_interval))

        # dump model regularly
        if update_cnt % self.dump_interval == 0 and update_cnt > 0:
            self.store_model(model, f"model_update_{update_cnt}")

        # estimate updates per second (running avg)
        self.step_times[0:-1] = self.step_times[1::]
        self.step_times[-1] = time.time() - self.after
        ups = 1.0 / self.step_times.mean()
        self.after = time.time()
        print("update %d @ %.1fups" % (np.mod(update_cnt, self.log_interval), ups), end="\r")
        sys.stdout.flush()

    def log_scalars(self, log_dict: dict, stage: str, step: int) -> None:
        if self.log_writer is not None:
            for key in log_dict:
                self.log_writer.add_scalar(f'{stage}/{key}', log_dict[key].item(), step)

    def store_model(self, model: models.Model, tag: str) -> None:
        if self.dump_dir is not None:
            print(f'Saving model with tag: {tag}')
            model.save_network(os.path.join(self.dump_dir, tag))

    def close(self) -> None:
        if self.log_writer is not None:
            self.log_writer.close()

    @staticmethod
    def setup_logger(args) -> Logger:

        time_stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        tr_set = os.path.basename(args.train_set)
        config_name = os.path.basename(args.game_config).split(".yaml")[0]
        user = getpass.getuser()
        exp_dir = args.model + "-" + args.net + "-" + tr_set + "-" + config_name + "_" + time_stamp + "-" + user

        args.experiment_directory = exp_dir
        args.dump_dir = None

        if args.no_log:
            log_writer = None
        else:

            # create model parameter directory
            args.dump_dir = os.path.join(args.dump_root, exp_dir)

            log_dir = os.path.join(args.log_root, args.experiment_directory)

            if not os.path.exists(args.log_root):
                os.makedirs(args.log_root)

            if not os.path.exists(args.dump_root):
                os.makedirs(args.dump_root)

            if not os.path.exists(args.dump_dir):
                os.mkdir(args.dump_dir)

            log_writer = SummaryWriter(log_dir=log_dir)

            text = ""
            arguments = np.sort([arg for arg in vars(args)])
            for arg in arguments:
                text += f"**{arg}:** {getattr(args, arg)}<br>"

            log_writer.add_text("run_config", text)
            log_writer.add_text("cmd", " ".join(sys.argv))

        return Logger(log_writer, log_gradients=args.log_gradients, log_interval=args.log_interval,
                      dump_dir=args.dump_dir, dump_interval=args.dump_interval)
