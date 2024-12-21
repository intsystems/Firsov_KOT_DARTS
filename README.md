## Abstract

The paper examines the problem of deep learning model selection with respect to target hardware. We propose a method for neural architecture search (NAS) that accounts for hardware constraints and model complexity. The complexity of the model is measured by the number of parameters, while the hardware constraints are represented by the overall latency of network operations on the target device. 

This approach builds upon Differentiable Architecture Search (DARTS), treating network structural parameters as functions of a complexity parameter, and extends it by incorporating latency-aware optimization inspired by FBNet. By introducing Gumbel-Softmax sampling and hypernetworks, we enable simultaneous optimization of multiple architectures across a range of complexity levels. This results in a single optimization process that identifies a family of architectures tailored to different computational budgets, reducing search time and resources.

---

## Introduction

Selecting an appropriate architecture for deep learning models is a crucial task that directly impacts model efficiency and performance. With deep learning continuing to push computational limits, researchers face the challenge of finding a balance between model complexity, accuracy, and resource consumption. Recent advances in Neural Architecture Search (NAS) techniques, such as Differentiable Architecture Search (DARTS), seek to automate this process by exploring large search spaces of possible network structures. However, these methods often struggle with high computational requirements and the need for architecture adjustments when model complexity or target hardware changes
One of the significant developments in NAS is the introduction of hardware-aware models. For example, FBNet~\cite{Wu_2019_CVPR} incorporates latency into the architecture search process, optimizing not only model performance but also hardware efficiency. This approach addresses the mismatch between FLOPs and actual hardware performance, a limitation of many prior NAS methods. FBNet achieves this through gradient-based optimization and Gumbel-Softmax sampling, which dramatically reduce the search costs while generating a family of hardware-optimized models.
% MnasNet

Similar ideas are used in ProxylessNAS, which solves the problem of high memory and computing costs by optimizing architectures directly on large tasks and target hardware platforms, without using proxy tasks. ProxylessNAS introduces a direct search engine (Reinforce sampling) for learning architectures on large datasets and simulates operation delays to account for hardware limitations. However, this approach requires careful calibration, as enhanced learning often suffers from high variance.

Building on these ideas, our work improves upon DARTS-CC
, a NAS approach that uses hypernetworks to control model complexity during architecture search. Unlike other methods that search for individual architectures at different complexity levels, DARTS-CC generates multiple architectures in a single optimization process. Inspired by FBNet, we extend DARTS-CC by integrating latency-aware optimization and replacing the scalar complexity parameter with a simplex-based representation of architecture choices. This enables simultaneous search for architectures optimized across multiple complexity and latency levels, further reducing NAS time, and ensuring deployability across diverse hardware environments.

Our contributions can be summarized as follows:
- We extend DARTS-CC by incorporating hardware latency into the optimization process. This enables the discovery of architectures that are not only accurate, but also efficient on target devices.
- We replace the scalar complexity parameter in DARTS-CC with a simplex-based representation, allowing the use of Gumbel-Softmax sampling for flexible architecture optimization.
- We propose a unified framework that combines latency-aware optimization, hypernetworks, and Gumbel-Softmax sampling to generate a family of architectures in a single optimization run.
- We validate our method on multiple datasets and hardware platforms, demonstrating improved efficiency and flexibility compared to existing NAS approaches.
