# Framework-Guided On-Policy Distillation

## Overview

本项目研究 **Framework-Guided On-Policy Distillation（框架引导的在线策略蒸馏）**。

以 **GKD（Generalized Knowledge Distillation，广义知识蒸馏）** 为基础，在 Student 进行 rollout（生成轨迹）前提供抽象的解题框架，引导 Student 生成结构更合理、更高质量的推理轨迹，并研究这种 rollout 改善能否进一步提升 OPD 的蒸馏效果。

核心思路：

```text
Vanilla OPD:
Question
   ↓
Student Rollout
   ↓
Teacher Token-level Supervision
   ↓
Update Student


Framework-Guided OPD:
Question
   ↓
Framework Teacher
   ↓
Reasoning Framework
   ↓
Student Rollout
   ↓
Teacher Token-level Supervision
   ↓
Update Student
```

Framework 只描述解题步骤和思考结构，不直接提供具体答案。

## Experiment

当前实验主要基于数学推理任务进行，对比：

- **Vanilla OPD（标准在线策略蒸馏）**
- **Framework-Guided OPD（框架引导在线策略蒸馏）**

基本实验流程：

```text
Generate Framework Data
        ↓
Train Framework Teacher
        ↓
Framework Generation
        ↓
Vanilla / Guided OPD Training
        ↓
Evaluation
        ↓
Comparison
```

主要关注：

- Student 最终任务准确率
- Student rollout 质量
- rollout 长度与截断情况
- Framework 对不同难度问题的影响
- Framework 对 OPD 训练过程的影响

## Key Factors

后续实验重点考察：

```text
max_steps / epochs
max_new_tokens
max_length
temperature
top_p
on-policy ratio (λ)
GKD divergence / β
learning rate
effective batch size
rollout number
```

其中重点研究 **Framework × On-Policy Rollout（框架 × 在线策略生成）** 的交互作用，而非单纯进行超参数搜索。

## Goal

核心研究问题：

> **Can abstract reasoning frameworks improve student-generated rollouts and thereby improve on-policy distillation?**

（抽象推理框架能否改善 Student 自生成的推理轨迹，并进一步提升在线策略蒸馏效果？）
