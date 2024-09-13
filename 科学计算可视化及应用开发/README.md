## 如何配置环境
> for MacOS(M2)

1. 安装 Xcode（未验证必要性，但要 Qt 的话必须得有）

    App Store 里搜索 Xcode 下载安装就行，实测只需要默认组件

2. 基于 Homebrew 安装 VTK

    ```shell
    brew install vtk
    ```

    非常慢，但目测把 qt 啥的都安装上了

## 如何构建

```shell
mkdir build
cmake ..
make
```

## 如何运行

```shell
./build/sample # sample 可以改成 project name
```

## 需要修改的


@ CMakeLists.txt

- line3: `project(sample)`，改个阳间点的
- line22: `add_executable(${PROJECT_NAME} sample.cpp)`，改成相应的 cpp 文件名
