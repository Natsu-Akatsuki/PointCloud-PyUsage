# GIF

有关GIF的转换、录制、剪辑、倍速工具

## [Gifski](https://gif.ski/)

### Install

```bash
$ wget -c https://gif.ski/gifski-1.10.0.zip
$ unzip gifski-1.10.0.zip
$ sudo dpkg -i linux/gifski_1.10.0_amd64.deb
```

### 将mp4转换为gif

```bash
# --fast-forward：倍率
# --quality：gif图质量
$ gifski --fps 30 --fast-forward 10 --width 320 -o <gif文件名> <mp4文件名>
```

## kdenlive

视频剪辑软件

### Install

```bash
$ sudo apt install kdenlive
```

## Imageio

修订Gif的速度

```python
import imageio

gif_original = '源文件名'
gif_speed_up = '导出文件名'
gif = imageio.mimread(gif_original)

imageio.mimsave(gif_speed_up, gif, fps=30) # 相关速度
```

