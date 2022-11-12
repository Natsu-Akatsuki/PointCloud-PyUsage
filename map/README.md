# README

对SLAM构建的地图进行处理

## 程序

|      模块名       |                 作用                 |
| :---------------: | :----------------------------------: |
| `cc-converter.py` | 将cc的`pcd`点云转换为通用的`pcd`文件 |
| `prepross_map.py` |   点云地图下采样、基于统计学的滤波   |

## 工具

### CloudCompare

#### [Install](http://www.cloudcompare.org/)

```bash
# 方法一：可以直接使用apt安装，但是部分apt版本不支持pcd点云文件的导入
# ubuntu22.04: 2.11
$ sudo apt install cloudcompare
# 方法二：使用snap安装
$ sudo snap install cloudcompare
# 切换至edge版本（其他可选：stable、beta）
$ sudo snap refresh --edge cloudcompare
```

#### Q&A

- [为什么cloudcompare没有撤销操作](http://www.danielgm.net/cc/forum/viewtopic.php?t=1257)
- [CloudCompare支持的文件格式](https://www.cloudcompare.org/doc/wiki/index.php?title=FILE_I/O)

#### Reference

- [官方视频教程](http://www.cloudcompare.org/tutorials.html)：包括剔除点云（仅支持2D裁剪）、配准（自动配准、交互式配准：自己选配准点）
- [官方说明文档](https://www.cloudcompare.org/doc/wiki/index.php/Main_Page)：[选点](https://www.cloudcompare.org/doc/wiki/index.php/Point_picking)
- [CloudCompare的Python版本](https://github.com/CloudCompare/CloudComPy)，需编译或者用conda

