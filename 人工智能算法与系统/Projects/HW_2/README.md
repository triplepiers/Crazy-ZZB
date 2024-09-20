# README

## 运行须知

- 由于源数据过大，此处暂不提供。

    请自行到 Mo 平台下载、放到 `./datasets` 下的相应位置，并按需修改 `main.ipynb` 中的 `path` 变量。

    也可以直接到 [这个页面](https://dgraph.xinye.com/dataset) 下载源数据（需要注册登陆orz）

- 请确保根路径下存在 `./results` 文件夹

- 如果版本不太兼容，可以移除 `./datasets/DGraph/processed` 文件夹

## Refs

- GraphSAGE: 

    - code: https://github.com/williamleif/GraphSAGE
    - paper: https://arxiv.org/abs/1706.02216

- GAT:

    - code: https://github.com/PetarV-/GAT
    - paper: https://arxiv.org/abs/1710.10903

- GATv2

    - code: https://github.com/DGraphXinye/DGraphFin_baseline（甚至是 BaseLine，但还是比 MLP 好点）

        ```bash
        python gnn_mini_batch.py --model gatv2_neighsampler --dataset DGraphFin --epochs 200 --runs 10 --device 0
        ```

    - paper: https://arxiv.org/abs/2105.14491

目标：在 Valid 上 > 0.74

- 利用NeighorSampler实现节点维度的mini-batch + GraphSAGE样例

     https://blog.csdn.net/weixin_39925939/article/details/121458145