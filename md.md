好的，你的项目是一个基于 XGBoost 的 IMU 动作识别项目，目前代码文件都是空的，data 目录也是空的。我来详细讲解如何用 IMU 6轴 200Hz 数据做动作识别数据集。

## IMU 6轴 200Hz 动作识别数据集制作完整指南
### 一、理解你的数据源
6轴 IMU 包含：

- 加速度计 (Accelerometer) : ax, ay, az（3轴，单位通常 m/s² 或 g）
- 陀螺仪 (Gyroscope) : gx, gy, gz（3轴，单位通常 °/s 或 rad/s）
200Hz 意味着每秒采样 200 次，每次得到 6 个通道的值。

### 二、数据采集阶段 1. 原始数据格式建议
每条采集记录应保存为 CSV，格式如下：

```
timestamp, ax, ay, az, gx, gy, gz, label
0.000, 0.12, -0.03, 9.81, 1.2, -0.5, 0.3, walking
0.005, 0.13, -0.02, 9.80, 1.3, -0.4, 0.2, walking
...
``` 2. 采集规范
要点 说明 采样率 固定 200Hz，时间间隔 5ms 动作类别 根据需求定义，如：walking, running, sitting, standing, jumping, falling 等 每次采集时长 建议每个动作持续 3~10 秒 采集人数 越多越好，至少 5~10 人 重复次数 每人每个动作至少重复 5~10 次 传感器位置 记录佩戴位置（手腕/腰部/脚踝等），不同位置数据差异很大
 3. 采集时的标注方式
有三种常见方式：

- 实时标注 ：采集时手动标记当前动作类别（最简单）
- 后标注 ：先录视频，事后根据视频标注（更准确）
- 半自动 ：按动作分段采集，每段文件名即标签
### 三、数据预处理阶段
这是做数据集的 核心步骤 ：
 1. 时间对齐与重采样
```
# 确保 200Hz 等间隔，如有丢包需插值
import pandas as pd
df = pd.read_csv('raw_data.csv')
df = df.set_index('timestamp')
df = df.resample('5ms').interpolate(method='linear')
``` 2. 滤波去噪
```
- 低通滤波：去除高频噪声，截止频率通常 20Hz（人体运动频率 < 20Hz）
- 可选：巴特沃斯滤波器（Butterworth），4阶，20Hz 截止
``` 3. 滑动窗口分割（最关键）
这是将连续时序数据转为监督学习样本的关键步骤：

```
参数选择：
├── 窗口长度 (window_size): 通常 1~3 秒
│   └── 200Hz × 2秒 = 400 个采样点/窗口
├── 滑动步长 (stride): 通常 50% 重叠
│   └── stride = window_size / 2 = 200 个采样点
└── 标签分配：窗口内占比最大的标签（或要求窗口内标签一致）
```
示意：

```
原始数据: [----walking----][---running---][---sitting---]
                    200Hz 连续采样

滑动窗口:
  窗口1: [0:400]     label=walking
  窗口2: [200:600]   label=walking  (50%重叠)
  窗口3: [400:800]   label=running
  ...
``` 4. 数据标准化
```
- Z-score 标准化: (x - mean) / std
- 或 Min-Max 归一化: (x - min) / (max - min)
- 注意：mean/std/min/max 应在训练集上计算，然后应用到验证/测试集
```
### 四、特征工程阶段
每个滑动窗口提取特征，将 (400, 6) 的时序数据压缩为固定长度的特征向量：
 时域特征（每个通道各算一遍，6通道 × N个特征）
特征 公式/说明 均值 (mean) 窗口内平均值 标准差 (std) 窗口内标准差 最大值 (max) 窗口内最大值 最小值 (min) 窗口内最小值 峰峰值 (range) max - min 中位数 (median) 窗口中位数 偏度 (skewness) 分布偏斜程度 峰度 (kurtosis) 分布尖锐程度 过零率 (zero crossing rate) 信号过零点次数 四分位距 (IQR) Q3 - Q1 信号能量 (energy) sum(x²) RMS sqrt(mean(x²))
 频域特征
特征 说明 主频率 (dominant frequency) FFT 幅值最大的频率 频谱能量 各频段能量分布 频谱熵 频谱的熵值
 跨通道特征
特征 说明 加速度幅值 sqrt(ax² + ay² + az²) 的统计量 角速度幅值 sqrt(gx² + gy² + gz²) 的统计量 通道间相关系数 ax 与 gx 的相关系数等

最终特征维度 ：6通道 × ~15个时域特征 + 6通道 × ~3个频域特征 + ~6个跨通道特征 ≈ 100+ 维特征向量

### 五、数据集存储格式 方案 A：处理后的 CSV（适合 XGBoost）
```
data/processed/
├── features.csv    # 每行一个窗口的特征向量 + 标签
└── metadata.json   # 记录窗口大小、步长、标准化参数等

features.csv 格式:
ax_mean, ax_std, ax_max, ..., gz_energy, acc_mag_mean, ..., label
0.12,    0.03,   0.45,   ..., 2.31,        9.85,        ..., walking
``` 方案 B：原始窗口数据（适合深度学习）
```
data/processed/
├── X_train.npy    # shape: (N_samples, 400, 6)
├── y_train.npy    # shape: (N_samples,)
├── X_test.npy
└── y_test.npy
```
### 六、数据集划分
```
划分原则：
├── 训练集 : 验证集 : 测试集 = 7 : 1 : 2  (或 8:1:1)
├── ⚠️ 按人划分 (Leave-One-Subject-Out)！不要按时间随机划分
│   └── 避免同一个人的数据同时出现在训练和测试集 → 数据泄露
└── 如果数据量少，可用 LOSO 交叉验证
```
### 七、完整数据流水线总结
```
原始 IMU 数据 (200Hz, 6轴)
    │
    ▼
[1] 时间对齐 & 重采样
    │
    ▼
[2] 低通滤波去噪 (Butterworth 20Hz)
    │
    ▼
[3] 滑动窗口分割 (2s窗口, 1s步长 → 每窗口400×6)
    │
    ▼
[4] 特征提取 (时域+频域+跨通道 → ~100维向量)
    │
    ▼
[5] 标准化 (Z-score, 训练集统计量)
    │
    ▼
[6] 数据集划分 (按人划分 train/val/test)
    │
    ▼
最终数据集: features.csv → 输入 XGBoost 训练
```
### 八、数据量参考
动作类别数 每类样本数(窗口) 总窗口数 建议采集人数 4~6 类 200~500 1000~3000 5~10 人 8~12 类 300~800 2400~9600 10~20 人

你需要我帮你把这些步骤落地为代码，填充到项目的 preprocess.py 、 features.py 、 config.py 等文件中吗？我可以根据你的具体动作类别和原始数据格式来写。

任务完成