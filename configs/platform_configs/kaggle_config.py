from configs.platform_configs.base_platform_config import BasePlatformConfig

class KaggleConfig(BasePlatformConfig):

    working_dir = '/kaggle/working/'
    shard_output_dir = f'{working_dir}sharded_data'
    train_dir = '/kaggle/input/waveform-inversion/train_samples'
    test_dir = '/kaggle/input/waveform-inversion/test'
