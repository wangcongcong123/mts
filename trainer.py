import os
import random
import shutil
import json
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
import torch

from sklearn.metrics import log_loss, roc_auc_score, classification_report, accuracy_score
import numpy as np
from utils import batch_to_device, count_dm_params,get_criterion

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

try:
    import wandb

    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn("W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.")
    else:
        _has_wandb = False if os.getenv("WANDB_DISABLED") else True
except ImportError:
    _has_wandb = False


def is_wandb_available():
    return _has_wandb




def set_seed(seed):
    logger.info("set random seed {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer():
    def __init__(self, train_data, model, dev_data=None, eval_every=-1, patience=200, loss_fn="bce",
                 train_batch_size=32, verbose=True, eval_on="loss", device="cpu", save_path=None, train_epochs=5,
                 keep_ck_num=3, lr=1e-2, eval_batch_size=64, seed=211,
                 use_wandb=False):
        set_seed(seed)
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)

        if len(os.listdir(save_path)) > 1:
            out = input(
                "Output directory ({}) already exists and is not empty, you wanna remove it before start? (y/n)".format(
                    save_path))
            if out.lower() == "y":
                shutil.rmtree(save_path)
                os.makedirs(save_path, exist_ok=True)
                # we need keep the vocab file
                train_data.save_vocab(save_path)
            else:
                raise ValueError("Output directory ({}) already exists and is not empty".format(save_path))
        self.tb_writer = SummaryWriter()
        self.eval_every = -1
        self.keep_ck_num = keep_ck_num
        self.train_data = train_data
        self.train_batch_size = train_batch_size
        self.train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
        self.set_logger(save_path)
        self.total_train_steps = len(self.train_dataloader) * train_epochs

        if verbose:
            logger.info(model)
            total_count, trainable_count, non_trainable_count = count_dm_params(model)
            logger.info(f'  Total params: {total_count}')
            logger.info(f'  Trainable params: {trainable_count}')
            logger.info(f'  Non-trainable params: {non_trainable_count}')
            logger.info(f"  There are {len(train_data)}  training examples")
            if dev_data != None:
                logger.info(f"  There are {len(dev_data)} examples for development")

        self.model = model.to(device)
        self.train_epochs = train_epochs
        self.device = device
        self.eval_batch_size = eval_batch_size
        self.dev_data = dev_data
        self.criterion = get_criterion(loss_fn)
        self.loss_fn = loss_fn

        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)  # use default
        self.lr = lr
        self.eval_every = eval_every
        self.save_path = save_path
        assert eval_on in ["log_loss", "auc", "loss", "accuracy"]
        self.eval_on = eval_on
        self.best_score = -float("inf")
        self.patience = patience
        self.no_improve_count = 0
        self.use_wandb = use_wandb
        self.hyperparams_logging = ["train_epochs", "eval_batch_size", "train_batch_size", "no_improve_count",
                                    "device", "patience", "save_path", "eval_on", "eval_every", "use_wandb", "loss_fn",
                                    "keep_ck_num", "lr"]
        self.hyperparam_dict = {key: self.__dict__[key] for key in self.hyperparams_logging}

        if is_wandb_available() and use_wandb:
            # keep track of model topology and gradients if is_wandb_available and args!=None
            wandb.init(project="deep_ctr", config=self.hyperparam_dict, name="_".join(save_path.split(os.path.sep)))
            wandb.watch((self.model), log_freq=max(100, eval_every))

    def set_logger(self, save_path):
        if save_path != None:
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh = logging.FileHandler(os.path.join(save_path, "log.out"), mode="a")
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    def train(self):
        global_step = 0
        running_loss = 0.0
        eval_steps = 0
        for epoch in trange(self.train_epochs):
            train_preds = []
            train_labels = []
            logger.info(f"start training epoch {epoch + 1}")
            logger.info(f"training using device={self.device}")
            logger.info("\n*************hyperparam_dict**********\n")
            logger.info(json.dumps(self.hyperparam_dict, indent=2))
            epoch_train_loss = 0.0

            pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))

            for step, (X, y) in (pbar):
                self.optimizer.zero_grad()
                self.model.train()

                features, labels = batch_to_device(X, y, device=self.device)
                if self.loss_fn == "ce":
                    labels = labels.long()
                # zero the parameter gradients

                outputs = self.model(features.float())
                loss = self.criterion(outputs, labels)
                # total_loss=loss + self.model.reg_loss# add reg_loss to avoid overfitting
                loss.backward()
                self.optimizer.step()
                pbar.set_description(
                    f"training epoch {epoch + 1}/{self.train_epochs} iter {step}: train loss {loss.item():.5f}. lr {self.lr:e}")

                train_preds.extend(outputs.tolist()) if self.model.task == "reg" else train_preds.extend(
                    torch.argmax(outputs, -1).tolist())

                train_labels.extend(labels.tolist())

                # print statistics
                running_loss += loss.item()
                epoch_train_loss += loss.item()
                eval_steps += 1
                if eval_steps == self.eval_every:
                    logger.info(
                        f'\n*****************[epoch: {epoch + 1}, global step: {global_step + 1}] eval training set based on eval_every={self.eval_every}***************')
                    train_eval_metrics = self.eval_train_during_training(train_labels, train_preds)
                    train_eval_metrics["train_loss"] = running_loss / eval_steps
                    logger.info(json.dumps(train_eval_metrics, indent=2))
                    self.tensorboard_logging(train_eval_metrics, global_step)

                    # wandb logging train
                    if is_wandb_available() and self.use_wandb:
                        wandb.log(train_eval_metrics, step=global_step)

                    running_loss = 0.0
                    eval_steps = 0

                    # evalute if self.dev_data != None
                    if self.dev_data is not None:
                        is_out_of_patience=self.evaluate_dev_data(epoch,global_step)
                        if is_out_of_patience:
                            logger.info(f"  run out of patience={self.patience} and save model before exit")
                            self.save_checkpoint(os.path.join(self.save_path, f"ck_{global_step + 1}"))
                            return 0

                    self.save_checkpoint(os.path.join(self.save_path, f"ck_{global_step + 1}"))
                global_step += 1

            logger.info(
                f'\n*****************[epoch: {epoch + 1}, global step: {global_step + 1}] eval training set at end of epoch***************')
            eval_metrics = self.eval_train_during_training(train_labels, train_preds)
            eval_metrics["train_loss"] = epoch_train_loss / len(self.train_dataloader)
            logger.info(json.dumps(eval_metrics, indent=2))

            if self.eval_every==-1:
                if self.dev_data is not None:
                    is_out_of_patience=self.evaluate_dev_data(epoch, global_step)
                    if is_out_of_patience:
                        logger.info(f"  run out of patience={self.patience} and save model before exit")
                        self.save_checkpoint(os.path.join(self.save_path, f"ck_{global_step + 1}"))
                        return 0
                    self.save_checkpoint(os.path.join(self.save_path, f"ck_{global_step + 1}"))
        self.tb_writer.close()

    def evaluate_dev_data(self,epoch,global_step):
        dev_eval_metrics = self.model.evaluate(self.dev_data, device=self.device,
                                               eval_batch_size=self.eval_batch_size,
                                               criterion_name=self.loss_fn)
        logger.info(
            f'\n*****************[epoch: {epoch + 1}, global step: {global_step + 1}] eval development set based on eval_every={self.eval_every}***************')
        if self.model.task == "cls":
            logger.info(classification_report(dev_eval_metrics.pop("groundtruths"),
                                              dev_eval_metrics.pop("preds")))
        else:
            dev_eval_metrics.pop("groundtruths")
            dev_eval_metrics.pop("preds")
        is_best_so_far, is_out_of_patience = self.check_best_and_patience(dev_eval_metrics)

        dev_eval_metrics = {"dev_" + key: value for key, value in dev_eval_metrics.items()}
        dev_eval_metrics[f"dev_best_score_for_{self.eval_on}"] = self.best_score
        logger.info(json.dumps(dev_eval_metrics, indent=2))
        self.tensorboard_logging(dev_eval_metrics, global_step)

        if is_best_so_far:
            logger.info("   save the model with best score so far")
            self.save_checkpoint(self.save_path)

        logger.info(f"  no_improve_count: {self.no_improve_count}")
        logger.info(f"  patience: {self.patience}")


        # wandb logging dev
        if is_wandb_available() and self.use_wandb:
            wandb.log(dev_eval_metrics, step=global_step)
        return is_out_of_patience

    def tensorboard_logging(self, metrics, step):
        for key, score in metrics.items():
            self.tb_writer.add_scalar(key, score, step)

    def eval_train_during_training(self, train_labels, train_preds):
        eval_metrics = {}
        if self.model.task == "reg":
            # train_log_loss = log_loss(train_labels, train_preds)
            # train_auc = roc_auc_score(train_labels, train_preds)
            # eval_metrics["train_log_loss"] = train_log_loss
            # eval_metrics["train_auc"] = train_auc
            pass
        elif self.model.task == "cls":
            logger.info(classification_report(train_labels, train_preds))
            train_accuracy_score = accuracy_score(train_labels, train_preds)
            eval_metrics["train_accuracy_score"] = train_accuracy_score
        return eval_metrics

    def save_checkpoint(self, save_path):
        if self.keep_ck_num != 0:
            checkpoints_dir_names = [(d, int(d.split("_")[-1])) for d in os.listdir(self.save_path) if
                                     os.path.isdir(os.path.join(self.save_path, d))]

            # check how many checkpoints already saved
            logger.info(f"   Check {len(checkpoints_dir_names)} checkpoints already saved")
            if len(checkpoints_dir_names) >= self.keep_ck_num:
                logger.info(
                    "   There are more than keep_ck_num as specified, then remove the oldest saved checkpoint")
                # there are more than keep_ck_num as specified
                # then remove the oldest saved checkpoint
                sorted_cks = sorted(checkpoints_dir_names, key=lambda k: k[1])
                removed_ck = os.path.join(self.save_path, sorted_cks[0][0])
                logger.info(f"  Remove checkpoint {removed_ck}")
                shutil.rmtree(removed_ck)

            logger.info(f"  Save checkpoint to {save_path}")
            self.model.save(save_path)
            self.save_hyperparams(save_path)

    def save_hyperparams(self, save_path):
        with open(os.path.join(save_path, 'hyperparam_dict.json'), 'w') as f:
            json.dump(self.hyperparam_dict, f, indent=2)
        logger.info(f"save model to path: {save_path}")

    def check_best_and_patience(self, eval_metrics):
        is_best_so_far, is_out_of_paitence = False, False
        assert self.eval_on in eval_metrics

        if "loss" in self.eval_on:
            if -eval_metrics[self.eval_on] > self.best_score:
                self.best_score = -eval_metrics[self.eval_on]
                is_best_so_far = True
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
        else:
            if eval_metrics[self.eval_on] > self.best_score:
                self.best_score = eval_metrics[self.eval_on]
                is_best_so_far = True
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
        if self.no_improve_count >= self.patience:
            is_out_of_paitence = True
        return is_best_so_far, is_out_of_paitence
