import json
import os

import torch


class Classifier:

    def __init__(self, model, learning_rate, weight_decay, optimizer='SGD', objective='cross-entropy',
                 gradient_clip=None, device=torch.device('cpu')):
        self.model = model
        # Set up optimizer
        if optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        elif optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=False)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        # Set up criterion
        if objective == 'bce':
            self.criterion = torch.nn.BCELoss()
        elif objective == 'cross-entropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif objective == 'bce-with-logit':
            self.criterion = torch.nn.BCEWithLogitsLoss
        else:
            raise ValueError(f"Unknown objective: {objective}")
        self.objective = objective
        self.gradient_clip = gradient_clip
        self.device = device
        self.model.to(self.device)

    def training_step(self, batch):
        raise NotImplementedError()

    def validation_step(self, batch, softmax=True):
        raise NotImplementedError()


def read_model_args(model_dir):
    with open(os.path.join(model_dir, 'args.json')) as f:
        args = json.load(f)
    return args
