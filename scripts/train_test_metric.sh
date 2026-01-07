
TRIAL=${1}
NET=${2}
mkdir checkpoints
mkdir checkpoints/${NET}_${TRIAL}
python ./train.py --use_gpu --net ${NET} --name ${NET}_${TRIAL} --nepoch 20 --nepoch_decay 20 # --monotonic_postprocessor
# python ./train.py --net ${NET} --name ${NET}_${TRIAL} # --monotonic_postprocessor
python ./test_dataset_model.py --use_gpu --net ${NET} --model_path ./checkpoints/${NET}_${TRIAL}/latest_net_.pth  
