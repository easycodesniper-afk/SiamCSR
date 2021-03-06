# SiamCSR

Testing and training code of paper ***HIGH SPEED AND ROBUST RGB-THERMAL TRACKING VIA DUAL ATTENTIVE STREAM SIAMESE NETWORK*** of The International Geoscience and Remote Sensing Symposium (IGARSS). 

- python3.7

- cuda9.2+pytorch1.7.1(not so strict)

- Ubuntu 16.04 or newer version

- install all requirements
******
Update on 04/06/2022, our paper has been accepted by IGARSS 2022. Paper will be published in three months.
******
Update on 06/04/2022, SiamCSR详细讲解已上传至bilibili弹幕网：[RGB-T跟踪算法SiamCSR代码讲解](https://www.bilibili.com/video/BV1PS4y1i7Dw?spm_id_from=333.999.0.0)
******
![image](https://user-images.githubusercontent.com/54308136/171970050-7b4523f3-df0c-47df-adb6-a2183877f408.png)

# Run demo

Please mkdir 'models' at root path and download snapshots and pretrain models from 
Baidu Cloud Drive:https://pan.baidu.com/s/1id8wCvru8cgTe6tZynpXzA password:yl6s 
or Google Drive:https://drive.google.com/drive/folders/1EWGAMNGd34FEbXgVobyvZY3A5nkcUPh-?usp=sharing

```py
python bin/my_demo.py to run demo.
```
![image](https://github.com/easycodesniper-afk/SiamCSR/blob/master/gif/rgb.gif)
![image](https://github.com/easycodesniper-afk/SiamCSR/blob/master/gif/t.gif)

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
