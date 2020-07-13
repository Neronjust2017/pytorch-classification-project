import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseEvaluater
from utils import *
from model.metric import *
from evaluater.utils import test_uncertainities

class EvaluaterVd(BaseEvaluater):
    """
    Evaluater class
    """
    def __init__(self, model, criterion, metric_ftns, config, test_data_loader):
        super().__init__(model, criterion, metric_ftns, config)
        self.config = config
        self.test_data_loader = test_data_loader
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

    def evaluate(self, samples=1000):
        """
        Evaluate after training procedure finished

        :return: A log that contains information about validation
        """
        Outputs = torch.zeros(self.test_data_loader.n_samples, self.model.num_classes, samples).to(self.device)
        targets = torch.zeros(self.test_data_loader.n_samples)

        self.model.eval()

        with torch.no_grad(): # torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度。
            start = 0
            for batch_idx, (data, target) in enumerate(self.test_data_loader):
                end = len(data) + start
                data, target = data.to(self.device), target.to(self.device)

                loss = 0
                outputs = torch.zeros(data.shape[0], self.model.num_classes, samples).to(self.device)

                if samples == 1:
                    out, _ = self.model(data)
                    loss = self.criterion(out, target)
                    outputs[:, :, 0] = out

                elif samples > 1:
                    mlpdw_cum = 0

                    for i in range(samples):
                        out, _ = self.model(data, sample=True)
                        mlpdw_i = self.criterion(out, target)
                        mlpdw_cum = mlpdw_cum + mlpdw_i
                        outputs[:, :, i] = out

                    mlpdw = mlpdw_cum / samples
                    loss = mlpdw

                Outputs[start:end, :, :] = outputs
                targets[start:end] = target
                start = end

                self.test_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.test_metrics.update(met.__name__, met(outputs, target, type="VD"))

        result = self.test_metrics.result()
        # print logged informations to the screen
        for key, value in result.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

        # self._visualization(Outputs, targets)
        test_uncertainities(Outputs, targets, self.model.num_classes, self.logger, save_path=str(self.result_dir))
