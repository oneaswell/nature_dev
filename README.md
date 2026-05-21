# Action Recognition with XGBoost

基于XGBoost的动作识别算法，通过可穿戴传感器时序数据进行动作分类。

## 项目结构
├── data/
│ ├── raw/ # 原始传感器数据
│ └── processed/ # 特征提取后数据
├── src/
│ ├── preprocess.py # 数据预处理（滤波、分割）
│ ├── features.py # 特征工程（时域/频域特征）
│ ├── train.py # 模型训练与评估
│ └── utils.py # 工具函数
├── models/ # 训练好的模型文件
├── config.py # 参数配置
└── requirements.txt