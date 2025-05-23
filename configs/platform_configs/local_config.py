from configs.platform_configs.base_platform_config import BasePlatformConfig

class LocalConfig(BasePlatformConfig):

    working_dir = './outputs/'
    shard_output_dir = f'{working_dir}sharded_data'
    train_dir = './inputs/train_samples'
    test_dir = './inputs/test'
