# SiamCSR

Testing and training code of paper ***High Speed and Robust RGB-Thermal Tracking via Attentive Dual-Stream Siamese Network*** of The International Geoscience and Remote Sensing Symposium (IGARSS). 

- python3.7

- cuda9.2+pytorch1.7.1(not so strict)

- Ubuntu 16.04 or newer version

- install all requirements

# Run demo

Please mkdir 'models' at root path and download snapshots and pretrain models from this url:https://pan.baidu.com/s/1id8wCvru8cgTe6tZynpXzA password:yl6s

```py
python bin/my_demo.py to run demo.
```

# Test

Please mkdir 'dataset' at root path and download GTOT, RGB-T234 and put them into dataset directory.

```py
python bin/my_test_rgbt.py \
        --model_path /path/to/snapshots \      #snapshot path
        --result_dir /path/to/results \        #results path
        --testing_dataset GTOT                 #testing benchmark
```

# Train
Please download LasHeR and put it into dataset directory.
```py
python bin/my_train.py
```
