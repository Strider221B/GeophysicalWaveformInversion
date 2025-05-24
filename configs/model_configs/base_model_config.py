import torch

from helpers.constants import Constants

class BaseModelConfig:

    model_prefix = None
    learning_rate = None
    weight_decay = None
    plot_every_n_epochs = None
    scheduler_factor = None
    scheduler_patience = None
    use_cuda = torch.cuda.is_available()
    device = torch.device(Constants.CUDA if use_cuda else Constants.CPU)
    autocast_dtype = None


    _n_epochs = None

    @classmethod
    def get_epochs(cls, trial_run: bool) -> int:
        if trial_run:
            return 1
        return cls._n_epochs
