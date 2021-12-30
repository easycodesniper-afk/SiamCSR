# SiamCSR

Testing and training code of paper [] of The International Geoscience and Remote Sensing Symposium (IGARSS) 

- python3.7

- cuda9.2+pytorch1.7.1(not so strict)

- Please mkdir 'dataset' at root path and download GTOT, RGB-T234, LasHeR and put them in dataset directory.

- Please mkdir 'models' at root path and download snapshots and pretrain models from this url:https://pan.baidu.com/s/1id8wCvru8cgTe6tZynpXzA password:yl6s

- python bin/my_demo.py to run demo.

- python bin/my_test_rgbt.py to test on all sequences of GTOT or RGB-T234.

- python bin/my_train.py to train.