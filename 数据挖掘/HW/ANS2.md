# HW2 - Let's LLM+X

## 1 Requirements

​	Considering and writing about utilizing Large Language Models (LLMs) in your research area (flexible, you may choose any reasearch area you are interested in).

- More than 500 words
- Include the following important parts:
  1. A brief introduction of studied problem.
  2. Motivation: how LLMs can benefit in solving this problem.
  3. The basic idea of method.
  4. To assess the effectiveness of LLM+X? Provide an outline of the experiments.

## 2 LLM + AQA

- A Brief Introduction

  ​	动作质量评估（Action Quality Assessment, AQA）是人体动作识别（Human Action Recognition）的子领域，旨在自动实现对输入视频中包含的动作质量进行评测。部分研究将其视为回归预测任务，为每个输入视频给出单独的质量得分；另一部分研究将其视为排序任务，为输入的系列视频给出质量排序结果。

  ​	然而，目前的研究大多采用以视频级表征为输入的端到端模型，在预测的可解释性上存在欠缺。此外，这类方法在处理较长的、包含多个子阶段的视频时表现欠佳，且缺乏过程性评价结果、难以为用户改进动作提供详细依据。

  ​	目前的模型大多数是针对具体下游任务设计的，在泛化性上存在不足。这类方法要求的大规模详细标注特定数据集，为数据收集工作带来了巨大挑战，

- Motivation

  ​        大模型具备丰富世界知识，可以为不同领域的视频提供统一的编码器，使得通用质量评估模型成为可能、提升AQA模型的泛化性。

  ​        大模型的强大视频理解能力使得视频数据的自动化标注成为可能，降低了垂直领域的数据收集处理成本。

  ​        大模型的多模态处理使得模型能够同时处理评论文本、图像流和背景音频三种模态的数据，从被传统方法忽视的文本和音频信息上挖掘潜在信息、提高模型准确性。

- Basic Idea

  ​       通用的可解释AQA模型主要包含两个阶段：

  1. 子阶段划分：借助LLM的视频理解能力自动对输入视频进行逐帧分类。
  2. 逐阶段质量评估：按照（1）中的分类结果，每次为LLM输入单阶段的视频片段，要求模型进行10分制的评分、并给出详细的评语。

- Outline of Experiments

  1. 验证LLM在视频分割上的有效性：

     ​        在多个不同领域（医疗、体育运动等）的、已经有逐帧动作类型标注的数据集上分别使用LLM和当前视频分割的SOTA方法（如多时域TCN）执行逐帧标注任务，对比两者的accuracy和F1-score。

  2. 验证LLM在视频质量分数回归预测上的有效性：

     ​        在多个不同领域、已经有各子阶段得分标注的数据集上分别使用LLM和已有的端到端SOTA方法进行阶段性动作质量评估，并将各阶段得分之和作为视频最终质量得分。比较基于LLM的方法与其他SOTA方法的均方误差和排序误差。

  3. 验证LLM提供得分解释的可信度与可用性：

     ​       （这部分一般都是做调研）将LLM给出的文本解释与视频片段呈现给对应领域专家、调研LLM生成的解释是否可信；同样的，将两样信息呈现给受众（医生或运动员）、调研LLM生成的文本解释是否能切实帮助他们改进自己的技术动作。
