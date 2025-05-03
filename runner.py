from helpers.file_handler import FileHandler
from helpers.plot_helper import PlotHelper
from models.model_factory import ModelFactory
from models.model_runner import ModelRunner

class Runner:

    @staticmethod
    def run():
        FileHandler.clean_up()
        FileHandler.shard_from_kaggle_data()
        _, dataloader_train, dataloader_validation = FileHandler.create_data_loaders_from_shards()
        model, optimizer, loss_criterion = ModelFactory.initialize_model_with_components()
        history = ModelRunner.train_model(dataloader_train, dataloader_validation, model, optimizer, loss_criterion)
        PlotHelper.plot_history(history)
        ModelRunner.predict_on_kaggle_test_data()
