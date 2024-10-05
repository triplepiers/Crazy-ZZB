# README

## 运行须知

- 由于源数据过大，此处暂不提供。

    请自行到 Mo 平台下载、放到 `./datasets` 下的相应位置，并按需修改 `main.ipynb` 中的 `path` 变量。

    ⚠️ 不可以直接到 [这个页面](https://dgraph.xinye.com/dataset) 下载源数据（需要注册登陆orz）

    - 破案了，两者连 Feature 维度都不一样（OJ 上甚至多了 3 个）

    - 我现在最无语的是他网站上给的 DGraphfin 编号和 OJ 上不太一样（至少 train_set_idx不一样）
  
        看起来是 random split 的 emmm

- 请确保根路径下存在 `./results` 文件夹

- 如果 pytorch 版本不太兼容，可以移除 `./datasets/DGraph/processed` 文件夹

## Refs

- GraphSAGE: 

    - code: https://github.com/williamleif/GraphSAGE
    - paper: https://arxiv.org/abs/1706.02216

- GAT:

    - code: https://github.com/PetarV-/GAT
    - paper: https://arxiv.org/abs/1710.10903

- GATv2

    有点太离谱了，俺的 M2 连续跑了 11h 之后把所有标签都判成了 '0'

    😂 马德更离谱的来了，这东西能到 0.775（虽然原生 MLP 都 0.721 了）=> GraphSAGE 白看了家人们

    - code: https://github.com/DGraphXinye/DGraphFin_baseline（甚至是 BaseLine，但还是比 MLP 好点）

        ```bash
        python gnn_mini_batch.py --model gatv2_neighsampler --dataset DGraphFin --epochs 200 --runs 10 --device 0
        ```

    - paper: https://arxiv.org/abs/2105.14491

目标：在 Valid 上 > 0.74

- 利用NeighorSampler实现节点维度的mini-batch + GraphSAGE样例

     https://blog.csdn.net/weixin_39925939/article/details/121458145