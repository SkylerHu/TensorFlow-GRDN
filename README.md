## 说明

GRDN(Grouped Residual Dense Network)来源于[NTIRE 2019图像降噪赛事](https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Abdelhamed_NTIRE_2019_Challenge_on_Real_Image_Denoising_Methods_and_Results_CVPRW_2019_paper.pdf)

GRDN算法在图像降噪sRGB赛道中以`PSNR 39.932`/`SSIM 0.973`的成绩取得了冠军。

[GRDN算法论文地址](https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Kim_GRDNGrouped_Residual_Dense_Network_for_Real_Image_Denoising_and_GAN-Based_CVPRW_2019_paper.pdf)

这里使用`TensorFlow`复现了GRDN算法，使用的数据集[SSID](https://abdokamel.github.io/sidd/)，可自行下载完整数据集。

## 环境依赖
- python==3.6.4
- pip==21.3.1
- tensorflow==1.13.1


## 执行示例：
```bash
python main.py --data_dir ./data/SIDD_Medium_Srgb/Data --run prepare
# 其中dataset_count值是prepare有输出结果
python main.py --data_dir ./data/SIDD_Medium_Srgb/Data --run --dataset_count 1785 train
python main.py --data_dir ./data/SIDD_Medium_Srgb/Test --run eval_file
python main.py --data_dir ./data/SIDD_Medium_Srgb/Test --run test
```
