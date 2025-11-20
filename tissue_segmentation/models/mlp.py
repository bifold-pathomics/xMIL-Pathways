import torch
import torch.nn as nn

from .utils import Classifier


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, hidden_depth, dropout=0, snn=False):
        """
        :param input_dim: (int) Dimension of the incoming feature vectors.
        :param hidden_dim: (int) Dimension of the hidden linear layers applied to the feature vectors.
        :param hidden_depth: (int) Number of hidden layers.
        :param dropout: (int) Fraction of neurons to drop on each hidden layer after the activation.
            Pass 0 to apply no dropout.
        :param snn: (bool) Whether the network should be a self-normalizing neural network (Klambauer et al. 2017).
            If so, SELU and AlphaDropout are used instead of ReLU and Dropout, and a custom weight initialization
            method is applied.
        """
        super(MLP, self).__init__()
        # Determine building blocks
        activation_cls = nn.SELU if snn else nn.ReLU
        dropout_cls = nn.AlphaDropout if snn else nn.Dropout
        # Setup network
        self.network = nn.Sequential()
        for layer in range(hidden_depth):
            in_dim = input_dim if layer == 0 else hidden_dim
            self.network.append(nn.Linear(in_dim, hidden_dim))
            self.network.append(activation_cls())
            self.network.append(dropout_cls(dropout))
        if snn:
            self._initialize_snn()

    def _initialize_snn(self):
        """
        Custom weight initialization method for SNNs. Implementation adapted from:
        https://github.com/bioinf-jku/SNNs/blob/master/Pytorch/SelfNormalizingNetworks_MLP_MNIST.ipynb
        """
        for param in self.network.parameters():
            # biases zero
            if len(param.shape) == 1:
                nn.init.constant_(param, 0)
            # others using lecun-normal initialization
            else:
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        return self.network.forward(x)

    def freeze_layers(self, num_layers):
        """
        Freezes the given number of layers, starting from the one closest to the input.

        :param num_layers: (int)
        """
        for layer in range(num_layers):
            for param in self.network[layer].parameters():
                param.requires_grad = False

    def unfreeze_layers(self):
        """
        Unfreezes all layers.
        """
        for param in self.parameters():
            param.requires_grad = True


class ClassificationMLP(MLP):

    def __init__(self, input_dim, hidden_dim, hidden_depth, num_classes, num_targets=1, dropout=0, snn=False):
        """
        :param num_classes: (int) Number of classes to predict, i.e., dimension of the output layer.
        """
        super(ClassificationMLP, self).__init__(input_dim, hidden_dim, hidden_depth, dropout, snn)
        in_dim = input_dim if hidden_depth == 0 else hidden_dim
        self.num_classes = num_classes
        self.num_targets = num_targets
        self.network.append(nn.Linear(in_dim, num_classes*num_targets))
        if snn:
            self._initialize_snn()


class BinaryMLPClassifier(Classifier):

    def training_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        features, targets = batch['features'].to(self.device), batch['targets'].to(self.device)
        preds = self.model(features)
        preds = preds.view(-1, self.model.num_classes, self.model.num_targets)
        loss = self.criterion(preds, targets)
        loss.backward()
        if self.gradient_clip is not None and self.gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.optimizer.step()
        return preds.detach(), targets.detach(), loss.detach()

    def validation_step(self, batch, softmax=True):
        self.model.eval()
        features, targets = batch['features'].to(self.device), batch['targets'].to(self.device)
        preds = self.model(features)
        preds = preds.view(-1, self.model.num_classes, self.model.num_targets)
        loss = self.criterion(preds, targets) if torch.all(targets >= 0) else torch.tensor(0.0)
        if softmax:
            preds = nn.functional.softmax(preds, dim=1)
        return preds.detach(), targets.detach(), loss.detach(), batch.get('sample_ids')
