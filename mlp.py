import json
import os

from torch import nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import log_loss, roc_auc_score, classification_report, accuracy_score
from utils import batch_to_device,get_criterion
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(self, feature_dim, label_num, hidden_units=[512, 218, 128], dropout_p=0.0, use_bn=False,
                 linear_act="relu", device="cpu", task="cls"):
        super(MLP, self).__init__()
        assert task in ["cls", "reg"], "use either cls or reg"
        self.config_keys = ["feature_dim", "label_num", "hidden_units", "dropout_p", "linear_act", "device", "task",
                            "use_bn"]
        self.feature_dim = feature_dim
        self.label_num = label_num
        self.device = device
        self.hidden_units = hidden_units
        self.task = task
        self.dropout_p = dropout_p
        self.linear_act = linear_act

        self.all_hidden_units = [feature_dim] + hidden_units

        self.linears = nn.ModuleList(
            [nn.Linear(self.all_hidden_units[i], self.all_hidden_units[i + 1]) for i in
             range(len(self.all_hidden_units) - 1)])

        self.use_bn = use_bn
        # self.ln=torch.nn.LayerNorm(hidden_units[-1])
        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(self.all_hidden_units[i + 1]) for i in range(len(self.all_hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [self.activation_layer(linear_act) for _ in
             range(len(self.all_hidden_units) - 1)])

        self.dropout = nn.Dropout(dropout_p)
        self.linear_output = nn.Linear(self.all_hidden_units[-1], label_num)

    def activation_layer(sefl, act_name):
        if isinstance(act_name, str):
            if act_name.lower() == 'sigmoid':
                act_layer = nn.Sigmoid()
            elif act_name.lower() == 'relu':
                act_layer = nn.ReLU(inplace=True)
            elif act_name.lower() == 'prelu':
                act_layer = nn.PReLU()
        elif issubclass(act_name, nn.Module):
            act_layer = act_name()
        else:
            raise NotImplementedError
        return act_layer

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def forward(self, X):
        deep_input = X
        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = self.activation_layers[i](fc)
            fc = self.dropout(fc)
            deep_input = fc
        # deep_output = self.linear_output(self.ln(deep_input))
        deep_output = self.linear_output(deep_input)
        return deep_output if self.task == "cls" else deep_output.squeeze()

    def predict(self, data, device="cuda", batch_size=64):
        preds = []
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        self.to(device)
        with torch.no_grad():
            for X, _ in tqdm(dataloader, desc="predicting"):
                features = batch_to_device(X, device=device)
                outputs = self(features)
                preds.extend(outputs.tolist())
        return preds

    def evaluate(self, data, device="cpu", eval_batch_size=64, criterion_name="ce"):
        criterion=get_criterion(criterion_name)
        eval_dataloader = DataLoader(data, batch_size=eval_batch_size)
        t_loss = 0.0
        steps = 0
        preds, groundtruths = [], []
        self.to(device)
        with torch.no_grad():
            for X, y in tqdm(eval_dataloader, desc="evaluating"):
                features, labels = batch_to_device(X, y, device=device)
                outputs = self(features.float())
                loss = criterion(outputs,
                                 labels.long() if self.task == "cls" else labels.float())  # for future version, if self.task == "cls" where criterion is cross entropy loss so it needs long labels
                t_loss += loss.item()
                steps += 1
                if self.task == "cls":
                    preds.extend(torch.argmax(outputs, -1).tolist())
                else:
                    preds.extend(outputs.squeeze().tolist())
                groundtruths.extend(labels.tolist())

        if self.task == "cls":
            logger.info(classification_report(groundtruths, preds))
            acc = accuracy_score(groundtruths, preds)
            return {"loss": t_loss / steps, "accuracy": acc, "preds": preds, "groundtruths": groundtruths}
        else:
            # lloss = log_loss(groundtruths, preds)
            # auc = roc_auc_score(groundtruths, preds)
            return {"loss": t_loss / steps, "preds": preds, "groundtruths": groundtruths}

    @classmethod
    def load_model(cls,load_path):
        if not os.path.isfile(os.path.join(load_path,'MLP_config.json')):
            raise ValueError(f"In the model path does not find MLP_config.json file, you may have not trained yet")
        with open(os.path.join(load_path, 'MLP_config.json')) as f:
            config = json.load(f)
        model = cls(**config)
        model.load_state_dict(torch.load(os.path.join(load_path, 'weights.pt')))
        return model.eval()

    def save(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        torch.save(self.state_dict(), os.path.join(save_path, 'weights.pt'))
        with open(os.path.join(save_path, f'{self.__class__.__name__}_config.json'), 'w') as f:
            json.dump(self.get_config_dict(), f, indent=2)
        logger.info(f"save model to path: {save_path}")
