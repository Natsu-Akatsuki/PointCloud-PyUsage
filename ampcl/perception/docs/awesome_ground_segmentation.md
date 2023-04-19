## Awesome Ground Segmentation
## 论文集

| 时间 |                             论文                             |            单位             |    关键词    |                             Code                             |
| :--: | :----------------------------------------------------------: | :-------------------------: | :----------: | :----------------------------------------------------------: |
| 1981 | Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography |      SRI International      |      —       |                              —                               |
| 2017 | Fast segmentation of 3D point clouds: A paradigm on LiDAR data for autonomous vehicle applications |        明尼苏达大学         |     PCA      | [YouTube](https://www.youtube.com/watch?v=7NNpvtdrHkU&ab_channel=DimitriosZermas)，复现版本[GitHub](https://github.com/chrise96/3D_Ground_Segmentation) |
| 2022 | GndNet: Fast Ground Plane Estimation and Point Cloud Segmentation for Autonomous Vehicles | 法国格勒诺布尔-阿尔卑斯大学 |     CNN      |      [GitHub](https://github.com/anshulpaigwar/GndNet)       |
| 2021 | [R-GPF] ERASOR: Egocentric Ratio of Pseudo Occupancy-based Dynamic Object Removal for Static 3D Point Cloud Map Building |       韩国科学技术院        |  去动态点云  |       [GitHub](https://github.com/LimHyungTae/ERASOR)        |
| 2022 | Patchwork: Concentric Zone-Based Region-Wise Ground Segmentation With Ground Likelihood Estimation Using a 3D LiDAR Sensor |       韩国科学技术院        |   扇区表征   |      [GitHub](https://github.com/LimHyungTae/patchwork)      |
| 2022 | Patchwork++: Fast and Robust Ground Segmentation Solving Partial Under-Segmentation Using 3D Point Cloud |       韩国科学技术院        |   扇区表征   |  [GitHub](https://github.com/url-kaist/patchwork-plusplus)   |
| 2022 | TRAVEL: Traversable Ground and Above-Ground Object Segmentation Using Graph Representation of 3D LiDAR Scans |       韩国科学技术院        |  可行驶区域  |        [Github](https://github.com/url-kaist/TRAVEL)         |
|  —   |                          Patchwork2                          |       韩国科学技术院        | Under Review |      [GitHub](https://github.com/url-kaist/Patchwork2)       |

## 发展方向

早期基于`RANSAC`的地面分割方案，假设地面是一个平面和认为地面点的数量在一帧点云中是最多的；但在实际的情况中，地面可能是具有一系列的坡度的

（`FPGF`）通过PCA分析来减少RANSAC迭代的次数，从而提高时间效率。RANSAC的方法是相对而言抗噪声的，但是需要较多次的迭代次数，才能找到较优的结果。




