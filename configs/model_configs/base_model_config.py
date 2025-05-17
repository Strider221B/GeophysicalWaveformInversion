class BaseModelConfig:

    model_prefix = None
    learning_rate = None
    weight_decay = None
    plot_every_n_epochs = None
    scheduler_factor = None
    scheduler_patience = None

    _n_epochs = None

    @classmethod
    def get_epochs(cls, trial_run: bool) -> int:
        if trial_run:
            return 1
        return cls._n_epochs
