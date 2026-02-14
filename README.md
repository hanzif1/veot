# 🤖 MA-VideoToT: A Multi-Agent Tree-of-Thought Framework for Video Reasoning

**MA-VideoToT** is an innovative Multi-Agent Tree-of-Thought framework designed to overcome the limitations of single-path reasoning in complex video understanding. By generating multiple reasoning paths in parallel and selecting the optimal one through a competitive scoring mechanism, it ensures robust alignment between logical deduction and visual evidence.

------

## 🌟 Key Features

- **Multi-Agent Collaboration**: Utilizes specialized **Planner**, **Grounder**, and **Verifier** agents to optimize the reasoning process via a Chain-of-Thought strategy.
- **Tree-of-Thought (ToT) Structure**: Explores multiple reasoning branches in parallel, preventing the propagation of early decision errors common in linear models.
- **Dynamic Pruning Mechanism**: Features a Verifier-driven pruning strategy that marginalizes low-consistency trajectories, maintaining a lean memory footprint while improving accuracy.
- **Superior Efficiency**: Achieves state-of-the-art results on a **2B parameter** backbone, proving that structural optimization can outperform massive model scaling.

------

## 🏗️ Architecture

The framework operates through a four-stage structured pipeline designed for high-level temporal and causal reasoning.

### 1. Planner Module

Generates a diverse set of thought trajectories $H = \{h_1, h_2, ..., h_k\}$ as a conditional probability distribution:

$$ P(h_{i}|V,Q)=\prod_{t=1}^{t}P(z_{t}|V,Q,z_{<t}) $$

### 2. Grounder Module

Bridges abstract reasoning with concrete visual evidence by identifying relevant temporal segments $S_i$:

$$s_{i}=\Phi(h_{i},V)$$

### 3. Verifier Module

Quantifies multimodal congruence between the reasoning path and the visual evidence using a verification score $w_i$:

$$w_{i}=\mathcal{V}(h_{i},s_{i},Q)$$

### 4. Answer Module

Selects the optimal trajectory $\tau^*$ and synthesizes the final response:

$$\tau^{*}=arg \max_{h_{i}\in \mathcal{T}}w_{i}$$

------

## 📊 Experimental Results

MA-VideoToT consistently outperforms state-of-the-art models across major benchmarks, particularly in complex causal and long-form reasoning tasks.

| **Method**             | **MVBench (Acc%)** | **NextQA (Acc%)** | **Long VideoBench (Acc%)** |
| ---------------------- | ------------------ | ----------------- | -------------------------- |
| LLaVA-7B               | 36.0               | -                 | 40.3                       |
| VideoChat2             | 51.1               | 68.6              | 41.2                       |
| VideoMind              | 48.8               | 66.6              | 48.8                       |
| **MA-VideoToT (Ours)** | **54.0**           | **76.2**          | **49.3**                   |

> **Note**: Ablation studies confirm that performance peaks at a tree width of **$K=3$**, balancing path diversity with evidence precision.

------

## 🚀 Quick Start

### Installation



```Bash
git clone https://github.com/anonymous/MA-VideoToT.git
cd MA-VideoToT
pip install -r requirements.txt
```

### Basic Inference



```Python
from ma_videotot import VideoToTAgent

# Initialize the 2B model
agent = VideoToTAgent(model_path="checkpoints/mav-2b", k=3)

# Run reasoning on a video
question = "What is the order of the letters on the table at the end?"
answer = agent.reason(video_path="demo.mp4", query=question)

print(f"Verified Answer: {answer}")
```
