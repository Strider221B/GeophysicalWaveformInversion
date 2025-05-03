# GeophysicalWaveformInversion
https://www.kaggle.com/competitions/waveform-inversion

Terminal command line steps for uploading code as a dataset:

rm -r ~/temp/kaggle_model_staging/geo_phy_wav_inv/*
cp -r ~/Git/GeophysicalWaveformInversion/configs/ ~/temp/kaggle_model_staging/geo_phy_wav_inv/configs
cp -r ~/Git/GeophysicalWaveformInversion/helpers/ ~/temp/kaggle_model_staging/geo_phy_wav_inv/helpers
cp -r ~/Git/GeophysicalWaveformInversion/models/ ~/temp/kaggle_model_staging/geo_phy_wav_inv/models
cp -r ~/Git/GeophysicalWaveformInversion/runner.py ~/temp/kaggle_model_staging/geo_phy_wav_inv/runner.py
cp -r ~/Git/GeophysicalWaveformInversion/dataset-metadata.json ~/temp/kaggle_model_staging/geo_phy_wav_inv/dataset-metadata.json

cd ~/temp/kaggle_model_staging/geo_phy_wav_inv
kaggle datasets version -p . -t -r zip -m "update"

kaggle datasets create -p . -t -r zip
