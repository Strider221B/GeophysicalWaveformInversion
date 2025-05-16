from configs.config import Config
from configs.model_configs.base_model_config import BaseModelConfig
from configs.model_configs.u_net_config import UNetConfig
from helpers.file_handler import FileHandler
from helpers.plot_helper import PlotHelper
from models.factories.model_factory import ModelFactory
from models.model_runner import ModelRunner

class Runner:

    @staticmethod
    def run(model_config: BaseModelConfig):
        Config.initialize_model_params_with(model_config)
        FileHandler.clean_up()
        FileHandler.shard_from_kaggle_data()
        _, dataloader_train, dataloader_validation = FileHandler.create_data_loaders_from_shards()
        model, optimizer, loss_criterion, scheduler = ModelFactory.get_model_with_components()
        history = ModelRunner.train_model(dataloader_train,
                                          dataloader_validation,
                                          model,
                                          optimizer,
                                          loss_criterion,
                                          scheduler)
        PlotHelper.plot_history(history)
        ModelRunner.predict_on_kaggle_test_data()

if __name__ == '__main__':
    Runner.run(UNetConfig)
