- 笑死，在 GitHub 上找到别人的 [src](https://github.com/QikaiXu/Robot-Maze-Solving) 了

    注意区分：24 年要求继承 `QRobot`，这位直接继承了 `TorchRobot` 在写

    => which means 我们需要参照 `TorchRobot` 的实现自己再搓一些成员函数

- 实际上提交的代码在 `main.py`，其中仅包含 DFS 的部分 + “自己” 实现的 DQN

    - 不知道为啥用 `mine: bool` 控制 QNetwork 的隐藏层在 OJ 上总是报错

        => 最后还是用了默认的 `(512, 512,)`

    - OJ 上给的 training epoch 对于本垃圾 bot 来说明显不足

        => 参照了最上面的 repo，在 `__init__()` 的时候自己先卷完