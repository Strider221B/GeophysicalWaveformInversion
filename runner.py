from configs.config import Config
from configs.model_configs.base_model_config import BaseModelConfig
# from configs.model_configs.hg_net_v2_config import HG_Net_V2_Config as model_config_to_use
from configs.model_configs.u_net_config import UNetConfig as model_config_to_use
from configs.platform_configs.base_platform_config import BasePlatformConfig
from configs.platform_configs.local_config import LocalConfig
from helpers.file_handler import FileHandler
from helpers.gpu_helper import GPUHelper
from helpers.plot_helper import PlotHelper
from models.factories.model_factory import ModelFactory
from models.model_runner import ModelRunner

class Runner:

    _initialized = False

    @classmethod
    def run(cls, model_config: BaseModelConfig, platform_config: BasePlatformConfig):
        '''
        Unfortunately the run method in it's ucrrent state cannot be used with multiple GPUs.
        We can use multiple GPUs with train but needs to be triggered via: torchrun --nproc_per_node=2 runner.py
        And inference must be trigerred via: python runner.py
        '''
        cls._initialize(model_config, platform_config)
        cls.create_shards(model_config, platform_config)
        cls.train_model(model_config, platform_config, False)
        cls.run_predictions(model_config, platform_config)

    @classmethod
    def run_predictions(cls,
                        model_config: BaseModelConfig,
                        platform_config: BasePlatformConfig):
        cls._initialize(model_config, platform_config)
        ModelRunner.predict_on_kaggle_test_data()
        GPUHelper.clean_all_memory(True)

    @classmethod
    def train_model(cls,
                    model_config: BaseModelConfig,
                    platform_config: BasePlatformConfig,
                    use_multiple_gpus):
        cls._initialize(model_config, platform_config)
        GPUHelper.run_on_multi_gpu_if_required(use_multiple_gpus)
        _, dataloader_train, dataloader_validation = FileHandler.create_data_loaders_from_shards()
        model, optimizer, loss_criterion, scheduler = ModelFactory.get_model_with_components()
        history = ModelRunner.train_model(dataloader_train,
                                          dataloader_validation,
                                          model,
                                          optimizer,
                                          loss_criterion,
                                          scheduler)
        GPUHelper.cleanup_multi_gpu_setup()
        PlotHelper.plot_history(history)
        GPUHelper.clean_all_memory(True)

    @classmethod
    def create_shards(cls,
                      model_config: BaseModelConfig,
                      platform_config: BasePlatformConfig):
        cls._initialize(model_config, platform_config)
        FileHandler.clean_up()
        FileHandler.shard_from_kaggle_data()

    @classmethod
    def _initialize(cls,
                    model_config: BaseModelConfig,
                    platform_config: BasePlatformConfig):
        if cls._initialized:
            return
        Config.initialize_params_with(model_config, platform_config, False)
        cls._initialized = True

if __name__ == '__main__':
    Config.trial_run = True
    Runner.run(model_config_to_use, LocalConfig, False)
    # Runner.create_shards(model_config_to_use, LocalConfig)
    # Runner.train_model(model_config_to_use, LocalConfig, False)
