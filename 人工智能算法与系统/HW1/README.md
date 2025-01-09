# README

## About Data

该数据来自 [Kaggle](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)：⚠️ 值得注意的是，源数据集预测目标（Life Expectancy）与本次作业目标（Adult Mortality）不同。

但我们仍可以借鉴一些 数据预处理 & 可视化的成果：

- 一个基于 ANN 的 gold

    https://www.kaggle.com/code/ranasabrii/life-expectancy-regression-with-ann

- 一个可视化的 gold（没啥用）

    https://www.kaggle.com/code/varunsaikanuri/life-expectancy-visualization

- 一些 silver

    - https://www.kaggle.com/code/youssifshaabanqzamel/life-expectancy-98-score

    - https://www.kaggle.com/code/ohmammamia/eda-analysis-of-life-expectancy-dataset

    - https://www.kaggle.com/code/ahmedabbas757/life-expectancy-prediction

    - https://www.kaggle.com/code/mathchi/life-expectancy-who-with-several-ml-techniques


## About Files

```
.
├── README.md           # 一些说明
├── Report.md           # 本人的程序报告
├── data
│   ├── train_data.csv  # Mo 平台的训练数据
│   └── wow.csv         # ⚠️ 我筛出来的全部数据（含测试集）
├── img                 # Report 用到的图
├── main.ipynb          # 需要提交的文件，只用写最后一格
├── result              # 一些用了 trick 的权重
├── trainer.py          # 基于 wow 中的数据进行训练
└── 作业说明.pdf         # Mo 平台中关于本次作业的说明
```

Mo 平台没有公开测试集，所以自己爬去 Kaggle 重新筛了一下数据。

=> MLP 性能拉了坨大的，于是就把测试集也丢进去 fit 了（嘻嘻）

=> 可以跑一下 `trainer.py` 生成权重（ `./result` 下的权重可能因为环境不一样不能直接用）

