# TabPFN-reproduction

本项目为 [TabPFN](https://github.com/automl/TabPFN) 的复现版本，旨在方便用户复现与理解 TabPFN 算法。核心代码包含数据创建、数据分桶（划分）、以及模型训练三个主要步骤。下方详细介绍每步操作流程，并结合原版 TabPFN 的信息，让你快速上手使用本仓库。

## 什么是 TabPFN？

**TabPFN**（Tabular Prior-Data Fitted Network）是由 AutoML 团队提出的一种面向结构化表格数据的神经网络，只需极少（甚至一步）训练即可实现高精度分类。其核心思想是用 Transformer 建模泛化过程，通过在神经网络中“模拟”大量小数据集的贝叶斯决策，实现极快、精准的表格数据分类。

详情见原论文与官方实现仓库：[TabPFN on GitHub](https://github.com/automl/TabPFN)

## 使用步骤

### 1. 配置文件

所有参数均在 `config/` 目录下的配置文件（如 `config.yaml` 或 `config.json`）中进行设置。请根据需要修改数据路径、分桶比例、模型参数等。

### 2. 数据创建

使用 `utils/create_data.py`，根据配置文件生成预处理数据。

```bash
python utils/create_data.py 
```

### 3. 获取分桶（训练/测试集划分）

使用 `get_split.py`，基于配置文件进行数据分桶：

```bash
python get_split.py
```


### 4. 一键运行

你也可以直接运行位于 `run/scripts/` 目录下的脚本，一步完成全部流程：

```bash
bash run/scripts/run.sh
```

该脚本会自动依次调用数据创建、分桶和训练过程，确保 config 配置路径正确。

## TabPFN 相关说明

- TabPFN 核心实现基于 Transformer 架构，可以极高速完成小样本表格数据的训练与推断。
- 该方法无需繁复调参，广泛适用于小型 tabular classification 任务，推荐用于开源竞赛首轮 baseline。
- 更多自定义参数（如网络深度、训练 epoch）请参考[官方 TabPFN 文档](https://github.com/automl/TabPFN#user-content-using-tabpfn)。

## 参考资料

- [TabPFN: Official Repo](https://github.com/automl/TabPFN)
- [TabPFN: One Transformer Predicts Tabular Data Like GPUs Predict Images (ICML 2023)](https://arxiv.org/abs/2207.01848)

---

## 联系作者

如有问题欢迎提 [Issues](https://github.com/rrrsj/TabPFN-reproduction/issues) 或邮件联系。
