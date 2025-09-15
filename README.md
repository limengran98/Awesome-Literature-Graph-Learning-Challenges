
# 📚 Awesome Literature: Graph Learning Challenges with LLMs

A curated list of recent research addressing fundamental challenges in graph learning with the assistance of large language models (LLMs). Papers are categorized by **challenge type** and **methodological approach**.

<p align="center">
  <img src="https://github.com/limengran98/Awesome-Literature-Graph-Learning-Challenges/blob/main/fig.jpg" width="80%">
</p>


*The four fundamental challenges emerge of real-world graph complexity: (1) **Incompleteness** in graphs, where nodes, edges, or attributes are missing, (2) **Imbalance** in graphs, where the distribution of nodes, edges, or labels is highly skewed, (3) **Cross-domain heterogeneity** in graphs, where graph data from different domains exhibit semantic and structural discrepancies, and (4) **Dynamic instability** in graphs, where graphs undergo dynamic changes in topology, attributes, or interactions over time.*

## 📰 News
- **2025-09-15**: Updated the **datasets, metrics, and tasks** sections with new entries and anchors.
- **2025-09-08**: Our survey paper was accepted by [*Expert Systems with Applications (ESWA)* 🎉](https://www.sciencedirect.com/science/article/abs/pii/S0957417425032580)  

## Table of Contents

- [Incompleteness in Graphs](#incompleteness-in-graphs)
  - [Robust Graph Learning](#robust-graph-learning)
  - [Few-shot Graph Learning](#few-shot-graph-learning)
  - [Knowledge Graph Completion](#knowledge-graph-completion)
  - [Datasets, Metrics, and Tasks (Incompleteness)](#datasets-metrics-and-tasks-incompleteness)
- [Imbalance in Graphs](#imbalance-in-graphs)
  - [Class-Imbalanced Graph Learning](#class-imbalanced-graph-learning)
  - [Structure-Imbalanced Graph Learning](#structure-imbalanced-graph-learning)
  - [Datasets, Metrics, and Tasks (Imbalance)](#datasets-metrics-and-tasks-imbalance)
- [Cross-Domain Heterogeneity in Graphs](#cross-domain-heterogeneity-in-graphs)
  - [Text-Attributed Graph Learning](#text-attributed-graph-learning)
  - [Multimodal Attributed Graph Learning](#multimodal-attributed-graph-learning)
  - [Structural Heterogeneous Graph Learning](#structural-heterogeneous-graph-learning)
  - [Datasets, Metrics, and Tasks (Cross-domain Heterogeneity)](#datasets-metrics-and-tasks-cross-domain-heterogeneity)
- [Dynamic Instability in Graphs](#dynamic-instability-in-graphs)
  - [LLMs for Querying and Reasoning](#querying-and-reasoning)
  - [LLMs for Generating and Updating](#generating-and-updating)
  - [LLMs for Evaluation and Application](#evaluation-and-application)
  - [Datasets, Metrics, and Tasks (Dynamic Instability)](#datasets-metrics-and-tasks-dynamic-instability)

---

##  Incompleteness in Graphs

> Graphs often suffer from missing node features, incomplete edges, or absent labels. These works tackle incompleteness via robust training, knowledge augmentation, or few-shot reasoning.

###  Robust Graph Learning
<!--
| Title | Authors | Venue & Year | Link |
|-------|---------|---------------|------|
| Spatiotemporal Pre-Trained Large Language Model for Forecasting With Missing Values | Le Fang, Wei Xiang, Shirui Pan, Flora D. Salim, Yi-Ping Phoebe Chen | IEEE Internet of Things Journal, 2025 | [Link](https://doi.org/10.1109/JIOT.2024.3524030) |
| LLM as Prompter: Low-resource Inductive Reasoning on Arbitrary Knowledge Graphs | Kai Wang, Yuwei Xu, Zhiyong Wu, Siqiang Luo | Findings of the Association for Computational Linguistics, 2024 | [Link](https://doi.org/10.18653/v1/2024.findings-acl.224) |
| On LLM-Enhanced Mixed-Type Data Imputation with High-Order Message Passing | Jianwei Wang, Kai Wang, Ying Zhang, Wenjie Zhang, Xiwei Xu, Xuemin Lin | arXiv preprint, 2025 | [Link](https://arxiv.org/abs/2501.02191) |
| Large language models as topological structure enhancers for text-attributed graphs | Shengyin Sun, Yuxiang Ren, Chen Ma, Xuecang Zhang | arXiv preprint, 2023 | [Link](https://doi.org/10.48550/arXiv.2311.14324) |
| Empower text-attributed graphs learning with large language models (LLMs) | Jianxiang Yu, Yuxiang Ren, Chenghua Gong, Jiaqi Tan, Xiang Li, Xuecang Zhang | arXiv preprint, 2023 | [Link](https://doi.org/10.48550/arXiv.2310.09872) |
| Label-free node classification on graphs with large language models (LLMs) | Zhikai Chen, Haitao Mao, Hongzhi Wen, Haoyu Han, Wei Jin, Haiyang Zhang, Hui Liu, Jiliang Tang | arXiv preprint, 2023 | [Link](https://arxiv.org/abs/2310.04668) |
| GraphLLM: Boosting graph reasoning ability of large language models | Ziwei Chai, Tianjie Zhang, Liang Wu, Kaiqiao Han, Xiaohai Hu, Xuanwen Huang, Yang Yang | arXiv preprint, 2023 | [Link](https://doi.org/10.48550/arXiv.2310.05845) |
-->

<details>
<summary><u><strong>) Spatiotemporal Pre-Trained Large Language Model for Forecasting With Missing Values (2025)</strong></u></summary>

**Authors:** Le Fang, Wei Xiang, Shirui Pan, Flora D. Salim, Yi-Ping Phoebe Chen  
**Venue & Year:** IEEE Internet of Things Journal, 2025  
**Link:** [https://doi.org/10.1109/JIOT.2024.3524030](https://doi.org/10.1109/JIOT.2024.3524030)
</details>
<details>
<summary><u><strong>) LLM as Prompter: Low-resource Inductive Reasoning on Arbitrary Knowledge Graphs (2024)</strong></u></summary>

**Authors:** Kai Wang, Yuwei Xu, Zhiyong Wu, Siqiang Luo  
**Venue & Year:** Findings of the Association for Computational Linguistics, 2024  
**Link:** [https://doi.org/10.18653/v1/2024.findings-acl.224](https://doi.org/10.18653/v1/2024.findings-acl.224)
</details>
<details>
<summary><u><strong>3) On LLM-Enhanced Mixed-Type Data Imputation with High-Order Message Passing (2025)</strong></u></summary>

**Authors:** Jianwei Wang, Kai Wang, Ying Zhang, Wenjie Zhang, Xiwei Xu, Xuemin Lin  
**Venue & Year:** arXiv preprint, 2025  
**Link:** [https://arxiv.org/abs/2501.02191](https://arxiv.org/abs/2501.02191)
</details>
<details>
<summary><u><strong>) Large language models as topological structure enhancers for text-attributed graphs (2023)</strong></u></summary>

**Authors:** Shengyin Sun, Yuxiang Ren, Chen Ma, Xuecang Zhang  
**Venue & Year:** arXiv preprint, 2023  
**Link:** [https://doi.org/10.48550/arXiv.2311.14324](https://doi.org/10.48550/arXiv.2311.14324)
</details>
<details>
<summary><u><strong>) Empower text-attributed graphs learning with large language models (LLMs) (2023)</strong></u></summary>

**Authors:** Jianxiang Yu, Yuxiang Ren, Chenghua Gong, Jiaqi Tan, Xiang Li, Xuecang Zhang  
**Venue & Year:** arXiv preprint, 2023  
**Link:** [https://doi.org/10.48550/arXiv.2310.09872](https://doi.org/10.48550/arXiv.2310.09872)
</details>
<details>
<summary><u><strong>) Label-free node classification on graphs with large language models (LLMs) (2023)</strong></u></summary>

**Authors:** Zhikai Chen, Haitao Mao, Hongzhi Wen, Haoyu Han, Wei Jin, Haiyang Zhang, Hui Liu, Jiliang Tang  
**Venue & Year:** arXiv preprint, 2023  
**Link:** [https://arxiv.org/abs/2310.04668](https://arxiv.org/abs/2310.04668)
</details>
<details>
<summary><u><strong>) GraphLLM: Boosting graph reasoning ability of large language models (2023)</strong></u></summary>

**Authors:** Ziwei Chai, Tianjie Zhang, Liang Wu, Kaiqiao Han, Xiaohai Hu, Xuanwen Huang, Yang Yang  
**Venue & Year:** arXiv preprint, 2023  
**Link:** [https://doi.org/10.48550/arXiv.2310.05845](https://doi.org/10.48550/arXiv.2310.05845)
</details>


---

###  Few-shot Graph Learning
<!--
| Title | Authors | Venue & Year | Link |
|-------|---------|---------------|------|
| LLM-Empowered Few-Shot Node Classification on Incomplete Graphs with Real Node Degrees | Yun Li, Yi Yang, Jiaqi Zhu, Hui Chen, Hongan Wang | Proceedings of the ACM International Conference on Information and Knowledge Management, 2024 | [Link](https://doi.org/10.1145/3627673.3679861) |
| LinkGPT: Teaching Large Language Models to Predict Missing Links | Zhongmou He, Jing Zhu, Shengyi Qian, Joyce Chai, Danai Koutra | arXiv preprint, 2024 | [Link](https://doi.org/10.48550/arXiv.2406.04640) |
| AnomalyLLM: Few-shot Anomaly Edge Detection for Dynamic Graphs using Large Language Models | Shuo Liu, Di Yao, Lanting Fang, Zhetao Li, Wenbin Li, Kaiyu Feng, XiaoWen Ji, Jingping Bi | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2405.07626) |
| HeGTa: Leveraging Heterogeneous Graph-enhanced Large Language Models for Few-shot Complex Table Understanding | Rihui Jin, Yu Li, Guilin Qi, Nan Hu, Yuan-Fang Li, Jiaoyan Chen, Jianan Wang, Yongrui Chen, Dehai Min, Sheng Bi | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2403.19723) |
| FlexKBQA: A Flexible LLM-powered Framework for Few-shot Knowledge Base Question Answering | Zhenyu Li, Sunqi Fan, Yu Gu, Xiuxing Li, Zhichao Duan, Bowen Dong, Ning Liu, Jianyong Wang | Proceedings of the AAAI Conference on Artificial Intelligence, 2024 | [Link](https://doi.org/10.1609/aaai.v38i17.29823) |
| Zero-shot Knowledge Graph Question Generation via Multi-agent LLMs and Small Models Synthesis | Runhao Zhao, Jiuyang Tang, Weixin Zeng, Ziyang Chen, Xiang Zhao | Proceedings of the ACM International Conference on Information and Knowledge Management, 2024 | [Link](https://doi.org/10.1145/3627673.3679805) |
-->
<details>
<summary><u><strong>LLM-Empowered Few-Shot Node Classification on Incomplete Graphs with Real Node Degrees (2024)</strong></u></summary>

**Authors:** Yun Li, Yi Yang, Jiaqi Zhu, Hui Chen, Hongan Wang  
**Venue & Year:** Proceedings of the ACM International Conference on Information and Knowledge Management, 2024  
**Link:** [https://doi.org/10.1145/3627673.3679861](https://doi.org/10.1145/3627673.3679861)
</details>
<details>
<summary><u><strong>LinkGPT: Teaching Large Language Models to Predict Missing Links (2024)</strong></u></summary>

**Authors:** Zhongmou He, Jing Zhu, Shengyi Qian, Joyce Chai, Danai Koutra  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://doi.org/10.48550/arXiv.2406.04640](https://doi.org/10.48550/arXiv.2406.04640)
</details>
<details>
<summary><u><strong>AnomalyLLM: Few-shot Anomaly Edge Detection for Dynamic Graphs using Large Language Models (2024)</strong></u></summary>

**Authors:** Shuo Liu, Di Yao, Lanting Fang, Zhetao Li, Wenbin Li, Kaiyu Feng, XiaoWen Ji, Jingping Bi  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2405.07626](https://arxiv.org/abs/2405.07626)
</details>
<details>
<summary><u><strong>HeGTa: Leveraging Heterogeneous Graph-enhanced Large Language Models for Few-shot Complex Table Understanding (2024)</strong></u></summary>

**Authors:** Rihui Jin, Yu Li, Guilin Qi, Nan Hu, Yuan-Fang Li, Jiaoyan Chen, Jianan Wang, Yongrui Chen, Dehai Min, Sheng Bi  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2403.19723](https://arxiv.org/abs/2403.19723)
</details>
<details>
<summary><u><strong>FlexKBQA: A Flexible LLM-powered Framework for Few-shot Knowledge Base Question Answering (2024)</strong></u></summary>

**Authors:** Zhenyu Li, Sunqi Fan, Yu Gu, Xiuxing Li, Zhichao Duan, Bowen Dong, Ning Liu, Jianyong Wang  
**Venue & Year:** Proceedings of the AAAI Conference on Artificial Intelligence, 2024  
**Link:** [https://doi.org/10.1609/aaai.v38i17.29823](https://doi.org/10.1609/aaai.v38i17.29823)
</details>
<details>
<summary><u><strong>Zero-shot Knowledge Graph Question Generation via Multi-agent LLMs and Small Models Synthesis (2024)</strong></u></summary>

**Authors:** Runhao Zhao, Jiuyang Tang, Weixin Zeng, Ziyang Chen, Xiang Zhao  
**Venue & Year:** Proceedings of the ACM International Conference on Information and Knowledge Management, 2024  
**Link:** [https://doi.org/10.1145/3627673.3679805](https://doi.org/10.1145/3627673.3679805)
</details>

---

###  Knowledge Graph Completion
<!--
| Title | Authors | Venue & Year | Link |
|-------|---------|---------------|------|
| GLTW: Joint Improved Graph Transformer and LLM via Three-Word Language for Knowledge Graph Completion | Kangyang Luo, Yuzhuo Bai, Cheng Gao, Shuzheng Si, Yingli Shen, Zhu Liu, Zhitong Wang, Wenhao Li, Yufei Huang | arXiv preprint, 2025 | [Link](https://www.arxiv.org/abs/2502.11471) |
| GS-KGC: A generative subgraph-based framework for knowledge graph completion with large language models | Rui Yang, Jiahao Zhu, Jianping Man, Hongze Liu, Li Fang, Yi Zhou | Information Fusion, 2025 | [Link](https://doi.org/10.1016/j.inffus.2024.102868) |
| In-Context Learning with Topological Information for LLM-Based Knowledge Graph Completion | Udari Madhushani Sehwag, Kassiani Papasotiriou, Jared Vann, Sumitra Ganesh | ICML 2024 Workshop on Structured Probabilistic Inference & Generative Modeling | [Link](https://openreview.net/forum?id=eUpH8AuVQa) |
| Making large language models perform better in knowledge graph completion | Yichi Zhang, Zhuo Chen, Lingbing Guo, Yajing Xu, Wen Zhang, Hlmruajun Chen | Proceedings of the 32nd ACM International Conference on Multimedia, 2024 | [Link](https://doi.org/10.1145/3664647.3681327) |
| LLM-based multi-level knowledge generation for few-shot knowledge graph completion | Qian Li, Zhuo Chen, Cheng Ji, Shiqi Jiang, Jianxin Li | Proceedings of the International Joint Conference on Artificial Intelligence, 2024 | [Link](https://www.ijcai.org/proceedings/2024/236) |
| Assessing LLMs Suitability for Knowledge Graph Completion | Vasile Ionut Remus Iga, Gheorghe Cosmin Silaghi | International Conference on Neural-Symbolic Learning and Reasoning, 2024 | [Link](https://doi.org/10.1007/978-3-031-71170-1_22) |
| Finetuning generative large language models with discrimination instructions for knowledge graph completion | Yang Liu, Xiaobin Tian, Zequn Sun, Wei Hu | International Semantic Web Conference, 2024 | [Link](https://doi.org/10.1007/978-3-031-77844-5_11) |
| Enhancing text-based knowledge graph completion with zero-shot large language models: A focus on semantic enhancement | Rui Yang, Jiahao Zhu, Jianping Man, Li Fang, Yi Zhou | Knowledge-Based Systems, 2024 | [Link](https://doi.org/10.1016/j.knosys.2024.112155) |
| Framing Few-Shot Knowledge Graph Completion with Large Language Models | Adrian MP Brasoveanu, Lyndon Nixon, Albert Weichselbraun, Arno Scharl | Joint Workshop Proceedings of the 5th International Workshop on Sem4Tra and SEMANTiCS, 2023 | [Link](https://ceur-ws.org/Vol-3510/paper_nlp_4.pdf) |
| Iterative zero-shot LLM prompting for knowledge graph construction | Salvatore Carta, Alessandro Giuliani, Leonardo Piano, Alessandro Sebastian Podda, Livio Pompianu, Sandro Gabriele Tiddia | arXiv preprint, 2023 | [Link](https://doi.org/10.48550/ARXIV.2307.01128) |
| Exploring large language models for knowledge graph completion | Liang Yao, Jiazhen Peng, Chengsheng Mao, Yuan Luo | arXiv preprint, 2023 | [Link](https://doi.org/10.48550/arXiv.2308.13916) |
| KICGPT: Large Language Model with Knowledge in Context for Knowledge Graph Completion | Yanbin Wei, Qiushi Huang, Yu Zhang, James Kwok | Findings of the Association for Computational Linguistics: EMNLP, 2023 | [Link](https://doi.org/10.18653/v1/2023.findings-emnlp.580) |
| Knowledge graph completion models are few-shot learners: An empirical study of relation labeling in e-commerce with LLMs | Jiao Chen, Luyi Ma, Xiaohan Li, Nikhil Thakurdesai, Jianpeng Xu, Jason HD Cho, Kaushiki Nag, Evren Korpeoglu, Sushant Kumar, Kannan Achan | arXiv preprint, 2023 | [Link](https://doi.org/10.48550/arXiv.2305.09858) |
| Generate-on-graph: Treat LLM as both agent and KG in incomplete knowledge graph question answering | Yao Xu, Shizhu He, Jiabei Chen, Zihao Wang, Yangqiu Song, Hanghang Tong, Guang Liu, Kang Liu, Jun Zhao | arXiv preprint, 2024 | [Link](https://doi.org/10.48550/arXiv.2307.01128) |
-->

<details>
<summary><u><strong>GLTW: Joint Improved Graph Transformer and LLM via Three-Word Language for Knowledge Graph Completion (2025)</strong></u></summary>

**Authors:** Kangyang Luo, Yuzhuo Bai, Cheng Gao, Shuzheng Si, Yingli Shen, Zhu Liu, Zhitong Wang, Wenhao Li, Yufei Huang  
**Venue & Year:** arXiv preprint, 2025  
**Link:** [https://www.arxiv.org/abs/2502.11471](https://www.arxiv.org/abs/2502.11471)
</details>
<details>
<summary><u><strong>GS-KGC: A generative subgraph-based framework for knowledge graph completion with large language models (2025)</strong></u></summary>

**Authors:** Rui Yang, Jiahao Zhu, Jianping Man, Hongze Liu, Li Fang, Yi Zhou  
**Venue & Year:** Information Fusion, 2025  
**Link:** [https://doi.org/10.1016/j.inffus.2024.102868](https://doi.org/10.1016/j.inffus.2024.102868)
</details>
<details>
<summary><u><strong>In-Context Learning with Topological Information for LLM-Based Knowledge Graph Completion (2024)</strong></u></summary>

**Authors:** Udari Madhushani Sehwag, Kassiani Papasotiriou, Jared Vann, Sumitra Ganesh  
**Venue & Year:** ICML 2024 Workshop on Structured Probabilistic Inference & Generative Modeling  
**Link:** [https://openreview.net/forum?id=eUpH8AuVQa](https://openreview.net/forum?id=eUpH8AuVQa)
</details>
<details>
<summary><u><strong>Making large language models perform better in knowledge graph completion (2024)</strong></u></summary>

**Authors:** Yichi Zhang, Zhuo Chen, Lingbing Guo, Yajing Xu, Wen Zhang, Hlmruajun Chen  
**Venue & Year:** Proceedings of the 32nd ACM International Conference on Multimedia, 2024  
**Link:** [https://doi.org/10.1145/3664647.3681327](https://doi.org/10.1145/3664647.3681327)
</details>
<details>
<summary><u><strong>LLM-based multi-level knowledge generation for few-shot knowledge graph completion (2024)</strong></u></summary>

**Authors:** Qian Li, Zhuo Chen, Cheng Ji, Shiqi Jiang, Jianxin Li  
**Venue & Year:** Proceedings of the International Joint Conference on Artificial Intelligence, 2024  
**Link:** [https://www.ijcai.org/proceedings/2024/236](https://www.ijcai.org/proceedings/2024/236)
</details>
<details>
<summary><u><strong>Assessing LLMs Suitability for Knowledge Graph Completion (2024)</strong></u></summary>

**Authors:** Vasile Ionut Remus Iga, Gheorghe Cosmin Silaghi  
**Venue & Year:** International Conference on Neural-Symbolic Learning and Reasoning, 2024  
**Link:** [https://doi.org/10.1007/978-3-031-71170-1_22](https://doi.org/10.1007/978-3-031-71170-1_22)
</details>
<details>
<summary><u><strong>Finetuning generative large language models with discrimination instructions for knowledge graph completion (2024)</strong></u></summary>

**Authors:** Yang Liu, Xiaobin Tian, Zequn Sun, Wei Hu  
**Venue & Year:** International Semantic Web Conference, 2024  
**Link:** [https://doi.org/10.1007/978-3-031-77844-5_11](https://doi.org/10.1007/978-3-031-77844-5_11)
</details>
<details>
<summary><u><strong>Enhancing text-based knowledge graph completion with zero-shot large language models: A focus on semantic enhancement (2024)</strong></u></summary>

**Authors:** Rui Yang, Jiahao Zhu, Jianping Man, Li Fang, Yi Zhou  
**Venue & Year:** Knowledge-Based Systems, 2024  
**Link:** [https://doi.org/10.1016/j.knosys.2024.112155](https://doi.org/10.1016/j.knosys.2024.112155)
</details>
<details>
<summary><u><strong>Framing Few-Shot Knowledge Graph Completion with Large Language Models (2023)</strong></u></summary>

**Authors:** Adrian MP Brasoveanu, Lyndon Nixon, Albert Weichselbraun, Arno Scharl  
**Venue & Year:** Joint Workshop Proceedings of the 5th International Workshop on Sem4Tra and SEMANTiCS, 2023  
**Link:** [https://ceur-ws.org/Vol-3510/paper_nlp_4.pdf](https://ceur-ws.org/Vol-3510/paper_nlp_4.pdf)
</details>
<details>
<summary><u><strong>Iterative zero-shot LLM prompting for knowledge graph construction (2023)</strong></u></summary>

**Authors:** Salvatore Carta, Alessandro Giuliani, Leonardo Piano, Alessandro Sebastian Podda, Livio Pompianu, Sandro Gabriele Tiddia  
**Venue & Year:** arXiv preprint, 2023  
**Link:** [https://doi.org/10.48550/ARXIV.2307.01128](https://doi.org/10.48550/ARXIV.2307.01128)
</details>
<details>
<summary><u><strong>Exploring large language models for knowledge graph completion (2023)</strong></u></summary>

**Authors:** Liang Yao, Jiazhen Peng, Chengsheng Mao, Yuan Luo  
**Venue & Year:** arXiv preprint, 2023  
**Link:** [https://doi.org/10.48550/arXiv.2308.13916](https://doi.org/10.48550/arXiv.2308.13916)
</details>
<details>
<summary><u><strong>KICGPT: Large Language Model with Knowledge in Context for Knowledge Graph Completion (2023)</strong></u></summary>

**Authors:** Yanbin Wei, Qiushi Huang, Yu Zhang, James Kwok  
**Venue & Year:** Findings of the Association for Computational Linguistics: EMNLP, 2023  
**Link:** [https://doi.org/10.18653/v1/2023.findings-emnlp.580](https://doi.org/10.18653/v1/2023.findings-emnlp.580)
</details>
<details>
<summary><u><strong>Knowledge graph completion models are few-shot learners: An empirical study of relation labeling in e-commerce with LLMs (2023)</strong></u></summary>

**Authors:** Jiao Chen, Luyi Ma, Xiaohan Li, Nikhil Thakurdesai, Jianpeng Xu, Jason HD Cho, Kaushiki Nag, Evren Korpeoglu, Sushant Kumar, Kannan Achan  
**Venue & Year:** arXiv preprint, 2023  
**Link:** [https://doi.org/10.48550/arXiv.2305.09858](https://doi.org/10.48550/arXiv.2305.09858)
</details>
<details>
<summary><u><strong>Generate-on-graph: Treat LLM as both agent and KG in incomplete knowledge graph question answering (2024)</strong></u></summary>

**Authors:** Yao Xu, Shizhu He, Jiabei Chen, Zihao Wang, Yangqiu Song, Hanghang Tong, Guang Liu, Kang Liu, Jun Zhao  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://doi.org/10.48550/arXiv.2307.01128](https://doi.org/10.48550/arXiv.2307.01128)
</details>


---

### Datasets, Metrics, and Tasks (Incompleteness)
#### 🔹 Robust Graph Learning
| Incompleteness | Method | Datasets | Metrics | Tasks |
|---------|--------|----------|---------|-------|
| 🟦 Node | **LLM4NG** | `Cora`, `PubMed`, `ogbn-arxiv` | 🎯 Accuracy | 🧩 Node Classification |
| 🟦 Node | **LLM-TAG** | `Cora`, `Citeseer`, `PubMed`, `Arxiv-2023` | 🎯 Accuracy | 🧩 Node Classification |
| 🟦 Node | **SPLLM** | `PeMS03`, `PeMS04`, `PeMS07` | 📉 MAE, RMSE, MAPE | ⏳ Spatiotemporal Forecasting |
| 🟥 Label | **LLMGNN** | `Cora`, `Citeseer`, `PubMed`, `ogbn-arxiv`, `ogbn-products`, `WikiCS` | 🎯 Accuracy | 🧩 Node Classification |
| 🟨 Mixed | **GraphLLM** | `Synthetic Data` | ✅ Exact Match Accuracy | 🔎 Graph Reasoning |
| 🟨 Mixed | **PROLINK** | `FB15k-237`, `Wikidata68K`, `NELL-995` | 📊 MRR, Hits@N | 🔗 Knowledge Graph Completion |
| 🟨 Mixed | **UnIMP** | `BG`, `ZO`, `PK`, `BK`, `CS`, `ST`, `PW`, `BY`, `RR`, `WM` | 📉 RMSE, MAE | 🔄 Data Imputation |

---

#### 🔹 Few-Shot Graph Learning
| Incompleteness | Method | Datasets | Metrics | Tasks |
|---------|--------|----------|---------|-------|
| 🟪 Structure | **LinkGPT** | `AmazonSports`, `Amazon-Clothing`, `MAG-Geology`, `MAG-Math` | 📊 MRR, Hits@N | 🔗 Link Prediction |
| 🟪 Structure | **AnomalyLLM** | `UCI Messages`, `Blogcatalog`, `T-Finance`, `T-Social` | 🚨 AUC | ⚠️ Anomaly Detection |
| 🟨 Mixed | **LLMDGCN** | `Cora`, `Citeseer`, `PubMed`, `Religion` | 🎯 Accuracy | 🧩 Node Classification |
| 🟨 Mixed | **HeGTa** | `IM-TQA`, `WCC`, `HiTab`, `WTQ`, `TabFact` | 📊 Macro-F1, Accuracy | 📑 Table Understanding |
| 🟨 Mixed | **FlexKBQA** | `GrailQA`, `WebQSP`, `KQA Pro` | ✅ Exact Match, F1, Accuracy | ❓ Knowledge Graph QA |
| 🟨 Mixed | **KGQG** | `WebQuestions`, `PathQuestions` | 📝 BLEU-4, ROUGE-L, Hits@N | ❓ Knowledge Graph QA |

---

#### 🔹 Knowledge Graph Completion
| Incompleteness | Method | Datasets | Metrics | Tasks |
|---------|--------|----------|---------|-------|
| 🟦 Node | **LLM-KGC** | `ILPC` | 📊 MRR, Hits@N | 🔗 Knowledge Graph Completion |
| 🟦 Node | **GS-KGC** | `WN18RR`, `FB15k-237`, `FB15k-237N`, `ICEWS14`, `ICEWS05-15` | 📊 Hits@N | 🔗 Knowledge Graph Completion |
| 🟦 Node | **GLTW** | `FB15k-237`, `WN18RR`, `Wikidata5M` | 📊 MRR, Hits@N | 🔗 Link Prediction |
| 🟥 Label | **KGs-LLM** | `Wikipedia` | 🎯 F1, Precision, Recall | 🏗️ Knowledge Graph Generation |
| 🟨 Mixed | **FSKG** | `WN18RR`, `FB15k-237` | 📊 MRR, Hits@N | 🔗 Knowledge Graph Completion |
| 🟨 Mixed | **KGLLM** | `WN11`, `FB13`, `WN18RR`, `YAGO3-10` | 🎯 Accuracy, MRR, Hits@N | 🔗 Link Prediction / KGC |
| 🟨 Mixed | **KICGPT** | `FB15k-237`, `WN18RR` | 📊 MRR, Hits@N | 🔗 Link Prediction |
| 🟨 Mixed | **RL-LLM** | `Electronics`, `Instacart` | 🎯 Precision, Recall, Accuracy | 🔗 Knowledge Graph Completion |
| 🟨 Mixed | **GoG** | `Synthetic Data` | 📊 Hits@N | ❓ Knowledge Graph QA |
| 🟨 Mixed | **KoPA** | `UMLS`, `CoDeX-S`, `FB15K-237N` | 🎯 F1, Precision, Recall, Accuracy | 🔗 Knowledge Graph Completion |
| 🟨 Mixed | **LLMKG** | `Templates Easy`, `Templates Hard` | 📏 Strict & Flexible Metrics | 🔗 Knowledge Graph Completion |
| 🟨 Mixed | **DIFT** | `WN18RR`, `FB15k-237` | 📊 MRR, Hits@N | 🔗 Link Prediction / KGC |
| 🟨 Mixed | **CP-KGC** | `WN18RR`, `FB15k-237`, `UMLS` | 📊 MRR, Hits@N | 🔗 Knowledge Graph Completion |
| 🟨 Mixed | **MuKDC** | `NELL`, `Wiki` | 📊 MRR, Hits@N | 🔗 Knowledge Graph Completion |


---

## Imbalance in Graphs

> Real-world graphs often exhibit skewed class distributions or unbalanced structural patterns, making training difficult and biased.

### Class-Imbalanced Graph Learning

<!--
| Title | Authors | Venue & Year | Link |
|-------|---------|---------------|------|
| LLM-Empowered Class Imbalanced Graph Prompt Learning for Online Drug Trafficking Detection | Tianyi Ma, Yiyue Qian, Zehong Wang, Zheyuan Zhang, Chuxu Zhang, Yanfang Ye | arXiv preprint, 2025 | [Link](https://arxiv.org/abs/2503.01900) |
| Large language model assisted fine-grained knowledge graph construction for robotic fault diagnosis | Xingming Liao, Chong Chen, Zhuowei Wang, Ying Liu, Tao Wang, Lianglun Cheng | Advanced Engineering Informatics, 2025 | [Link](https://doi.org/10.1016/j.aei.2025.103134) |
| Low-resource knowledge graph completion based on knowledge distillation driven by large language models | Wenlong Hou, Weidong Zhao, Ning Jia, Xianhui Liu | Applied Soft Computing, 2025 | [Link](https://doi.org/10.1016/j.asoc.2024.112622) |
| Empowering graph neural network-based computational drug repositioning with large language model-inferred knowledge representation | Yaowen Gu, Zidu Xu, Carl Yang | Interdisciplinary Sciences: Computational Life Sciences, 2024 | [Link](https://doi.org/10.1007/s12539-024-00654-7) |
| Cost-Effective Label-free Node Classification with LLMs | Taiyan Zhang, Renchi Yang, Mingyu Yan, Xiaochun Ye, Dongrui Fan, Yurui Lai | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2412.11983) |
| Enhancing student performance prediction on learnersourced questions with SGNN-LLM synergy | Lin Ni, Sijie Wang, Zeyu Zhang, Xiaoxuan Li, Xianda Zheng, Paul Denny, Jiamou Liu | Proceedings of the AAAI Conference on Artificial Intelligence, 2024 | [Link](https://doi.org/10.1609/aaai.v38i21.30370) |
| Depression detection in clinical interviews with LLM-empowered structural element graph | Zhuang Chen, Jiawen Deng, Jinfeng Zhou, Jincenzi Wu, Tieyun Qian, Minlie Huang | Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2024 | [Link](https://aclanthology.org/2024.naacl-long.452/) |
| Fine-grainedly Synthesize Streaming Data Based On Large Language Models With Graph Structure Understanding For Data Sparsity | Xin Zhang, Linhai Zhang, Deyu Zhou, Guoqiang Xu | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2403.06139) |
| Large Language Model-based Augmentation for Imbalanced Node Classification on Text-Attributed Graphs | Leyao Wang, Yu Wang, Bo Ni, Yuying Zhao, Tyler Derr | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2410.16882) |
| Distilling large language models for text-attributed graph learning | Bo Pan, Zheng Zhang, Yifei Zhang, Yuntong Hu, Liang Zhao | Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, 2024 | [Link](https://doi.org/10.1145/3627673.3679830) |
| KICGPT: Large Language Model with Knowledge in Context for Knowledge Graph Completion | Yanbin Wei, Qiushi Huang, Yu Zhang, James Kwok | Findings of the Association for Computational Linguistics: EMNLP, 2023 | [Link](https://doi.org/10.18653/v1/2023.findings-emnlp.580) |
| Label-free node classification on graphs with large language models (LLMs) | Zhikai Chen, Haitao Mao, Hongzhi Wen, Haoyu Han, Wei Jin, Haiyang Zhang, Hui Liu, Jiliang Tang | arXiv preprint, 2023 | [Link](https://arxiv.org/abs/2310.04668) |
| Augmenting low-resource text classification with graph-grounded pre-training and prompting | Zhihao Wen, Yuan Fang | Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2023 | [Link](https://doi.org/10.1145/3539618.3591641) |
| Leveraging Large Language Models for Node Generation in Few-Shot Learning on Text-Attributed Graphs | Jianxiang Yu, Yuxiang Ren, Chenghua Gong, Jiaqi Tan, Xiang Li, Xuecang Zhang | arXiv preprints, 2023 | [Link](https://arxiv.org/abs/2310.09872) |
| LKPNR: LLM and KG for Personalized News Recommendation Framework | Xie Runfeng, Cui Xiangyang, Yan Zhou, Wang Xin, Xuan Zhanwei, Zhang Kai, et al. | arXiv preprint, 2023 | [Link](https://arxiv.org/abs/2308.12028) |
-->

<details>
<summary><u><strong>LLM-Empowered Class Imbalanced Graph Prompt Learning for Online Drug Trafficking Detection (2025)</strong></u></summary>

**Authors:** Tianyi Ma, Yiyue Qian, Zehong Wang, Zheyuan Zhang, Chuxu Zhang, Yanfang Ye  
**Venue & Year:** arXiv preprint, 2025  
**Link:** [https://arxiv.org/abs/2503.01900](https://arxiv.org/abs/2503.01900)
</details>
<details>
<summary><u><strong>Large language model assisted fine-grained knowledge graph construction for robotic fault diagnosis (2025)</strong></u></summary>

**Authors:** Xingming Liao, Chong Chen, Zhuowei Wang, Ying Liu, Tao Wang, Lianglun Cheng  
**Venue & Year:** Advanced Engineering Informatics, 2025  
**Link:** [https://doi.org/10.1016/j.aei.2025.103134](https://doi.org/10.1016/j.aei.2025.103134)
</details>
<details>
<summary><u><strong>Low-resource knowledge graph completion based on knowledge distillation driven by large language models (2025)</strong></u></summary>

**Authors:** Wenlong Hou, Weidong Zhao, Ning Jia, Xianhui Liu  
**Venue & Year:** Applied Soft Computing, 2025  
**Link:** [https://doi.org/10.1016/j.asoc.2024.112622](https://doi.org/10.1016/j.asoc.2024.112622)
</details>
<details>
<summary><u><strong>Empowering graph neural network-based computational drug repositioning with large language model-inferred knowledge representation (2024)</strong></u></summary>

**Authors:** Yaowen Gu, Zidu Xu, Carl Yang  
**Venue & Year:** Interdisciplinary Sciences: Computational Life Sciences, 2024  
**Link:** [https://doi.org/10.1007/s12539-024-00654-7](https://doi.org/10.1007/s12539-024-00654-7)
</details>
<details>
<summary><u><strong>Cost-Effective Label-free Node Classification with LLMs (2024)</strong></u></summary>

**Authors:** Taiyan Zhang, Renchi Yang, Mingyu Yan, Xiaochun Ye, Dongrui Fan, Yurui Lai  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2412.11983](https://arxiv.org/abs/2412.11983)
</details>
<details>
<summary><u><strong>Enhancing student performance prediction on learnersourced questions with SGNN-LLM synergy (2024)</strong></u></summary>

**Authors:** Lin Ni, Sijie Wang, Zeyu Zhang, Xiaoxuan Li, Xianda Zheng, Paul Denny, Jiamou Liu  
**Venue & Year:** Proceedings of the AAAI Conference on Artificial Intelligence, 2024  
**Link:** [https://doi.org/10.1609/aaai.v38i21.30370](https://doi.org/10.1609/aaai.v38i21.30370)
</details>
<details>
<summary><u><strong>Depression detection in clinical interviews with LLM-empowered structural element graph (2024)</strong></u></summary>

**Authors:** Zhuang Chen, Jiawen Deng, Jinfeng Zhou, Jincenzi Wu, Tieyun Qian, Minlie Huang  
**Venue & Year:** Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2024  
**Link:** [https://aclanthology.org/2024.naacl-long.452/](https://aclanthology.org/2024.naacl-long.452/)
</details>
<details>
<summary><u><strong>Fine-grainedly Synthesize Streaming Data Based On Large Language Models With Graph Structure Understanding For Data Sparsity (2024)</strong></u></summary>

**Authors:** Xin Zhang, Linhai Zhang, Deyu Zhou, Guoqiang Xu  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2403.06139](https://arxiv.org/abs/2403.06139)
</details>
<details>
<summary><u><strong>Large Language Model-based Augmentation for Imbalanced Node Classification on Text-Attributed Graphs (2024)</strong></u></summary>

**Authors:** Leyao Wang, Yu Wang, Bo Ni, Yuying Zhao, Tyler Derr  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2410.16882](https://arxiv.org/abs/2410.16882)
</details>
<details>
<summary><u><strong>Distilling large language models for text-attributed graph learning (2024)</strong></u></summary>

**Authors:** Bo Pan, Zheng Zhang, Yifei Zhang, Yuntong Hu, Liang Zhao  
**Venue & Year:** Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, 2024  
**Link:** [https://doi.org/10.1145/3627673.3679830](https://doi.org/10.1145/3627673.3679830)
</details>
<details>
<summary><u><strong>KICGPT: Large Language Model with Knowledge in Context for Knowledge Graph Completion (2023)</strong></u></summary>

**Authors:** Yanbin Wei, Qiushi Huang, Yu Zhang, James Kwok  
**Venue & Year:** Findings of the Association for Computational Linguistics: EMNLP, 2023  
**Link:** [https://doi.org/10.18653/v1/2023.findings-emnlp.580](https://doi.org/10.18653/v1/2023.findings-emnlp.580)
</details>
<details>
<summary><u><strong>Label-free node classification on graphs with large language models (LLMs) (2023)</strong></u></summary>

**Authors:** Zhikai Chen, Haitao Mao, Hongzhi Wen, Haoyu Han, Wei Jin, Haiyang Zhang, Hui Liu, Jiliang Tang  
**Venue & Year:** arXiv preprint, 2023  
**Link:** [https://arxiv.org/abs/2310.04668](https://arxiv.org/abs/2310.04668)
</details>
<details>
<summary><u><strong>Augmenting low-resource text classification with graph-grounded pre-training and prompting (2023)</strong></u></summary>

**Authors:** Zhihao Wen, Yuan Fang  
**Venue & Year:** Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2023  
**Link:** [https://doi.org/10.1145/3539618.3591641](https://doi.org/10.1145/3539618.3591641)
</details>
<details>
<summary><u><strong>Leveraging Large Language Models for Node Generation in Few-Shot Learning on Text-Attributed Graphs (2023)</strong></u></summary>

**Authors:** Jianxiang Yu, Yuxiang Ren, Chenghua Gong, Jiaqi Tan, Xiang Li, Xuecang Zhang  
**Venue & Year:** arXiv preprints, 2023  
**Link:** [https://arxiv.org/abs/2310.09872](https://arxiv.org/abs/2310.09872)
</details>
<details>
<summary><u><strong>LKPNR: LLM and KG for Personalized News Recommendation Framework (2023)</strong></u></summary>

**Authors:** Xie Runfeng, Cui Xiangyang, Yan Zhou, Wang Xin, Xuan Zhanwei, Zhang Kai, et al.  
**Venue & Year:** arXiv preprint, 2023  
**Link:** [https://arxiv.org/abs/2308.12028](https://arxiv.org/abs/2308.12028)
</details>

---


###  Structure-Imbalanced Graph Learning
<!--
| Title | Authors | Venue & Year | Link |
|-------|---------|---------------|------|
| Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks? | Zhongjian Zhang, Xiao Wang, Huichi Zhou, Yue Yu, Mengmei Zhang, Cheng Yang, Chuan Shi | arXiv preprint, 2024 | [Link](https://doi.org/10.48550/arXiv.2408.08685) |
| Subgraph-Aware Training of Language Models for Knowledge Graph Completion Using Structure-Aware Contrastive Learning | Youmin Ko, Hyemin Yang, Taeuk Kim, Hyunjoon Kim | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2407.12703) |
| Multi-perspective improvement of knowledge graph completion with large language models | Derong Xu, Ziheng Zhang, Zhenxi Lin, Xian Wu, Zhihong Zhu, Tong Xu, Xiangyu Zhao, Yefeng Zheng, Enhong Chen | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2403.01972) |
| Graphedit: Large language models for graph structure learning | Zirui Guo, Lianghao Xia, Yanhua Yu, Yuling Wang, Zixuan Yang, Wei Wei, Liang Pang, Tat-Seng Chua, Chao Huang | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2402.15183) |
-->
<details>
<summary><u><strong>Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks? (2024)</strong></u></summary>

**Authors:** Zhongjian Zhang, Xiao Wang, Huichi Zhou, Yue Yu, Mengmei Zhang, Cheng Yang, Chuan Shi  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://doi.org/10.48550/arXiv.2408.08685](https://doi.org/10.48550/arXiv.2408.08685)
</details>
<details>
<summary><u><strong>Subgraph-Aware Training of Language Models for Knowledge Graph Completion Using Structure-Aware Contrastive Learning (2024)</strong></u></summary>

**Authors:** Youmin Ko, Hyemin Yang, Taeuk Kim, Hyunjoon Kim  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2407.12703](https://arxiv.org/abs/2407.12703)
</details>
<details>
<summary><u><strong>Multi-perspective improvement of knowledge graph completion with large language models (2024)</strong></u></summary>

**Authors:** Derong Xu, Ziheng Zhang, Zhenxi Lin, Xian Wu, Zhihong Zhu, Tong Xu, Xiangyu Zhao, Yefeng Zheng, Enhong Chen  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2403.01972](https://arxiv.org/abs/2403.01972)
</details>
<details>
<summary><u><strong>Graphedit: Large language models for graph structure learning (2024)</strong></u></summary>

**Authors:** Zirui Guo, Lianghao Xia, Yanhua Yu, Yuling Wang, Zixuan Yang, Wei Wei, Liang Pang, Tat-Seng Chua, Chao Huang  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2402.15183](https://arxiv.org/abs/2402.15183)
</details>

---

### Datasets, Metrics, and Tasks (Imbalance)

#### 🔹 Class Imbalance Graph Learning – Node Classification
| Tasks | Method | Datasets | Metrics | Downstream |
|-------|--------|----------|---------|-------------|
| 🧩 Node Classification | **LLM4NG** | `Cora`, `PubMed`, `ogbn-arxiv` | 🎯 Accuracy | 🪄 Few-shot Node Classification |
| 🧩 Node Classification | **LLM-GNN** | `Cora`, `Citeseer`, `PubMed`, `ogbn-arxiv`, `ogbn-products`, `WikiCS` | 🎯 Accuracy | 🏷️ Label-free Node Classification |
| 🧩 Node Classification | **G2P2** | `Cora`, `Amazon (Art, Industrial, Music Instruments)` | 🎯 Accuracy, 📊 Macro-F1 | 🌱 Zero-/Few-shot Low-resource Text Classification |
| 🧩 Node Classification | **LA-TAG** | `Cora`, `PubMed`, `Photo`, `Computer`, `Children` | 🎯 Accuracy, 📊 Macro-F1 | 🌱 Zero-/Few-shot Low-resource Text Classification |
| 🧩 Node Classification | **GSS-Net** | `Amazon (Magazine Subscriptions, Appliances, Gift Cards)` | 🎯 Accuracy, ✅ Precision, 🔍 Recall, 📊 F1, 📉 MSE, RMSE, MAE | 🛒 Sentiment on E-commerce Reviews |
| 🧩 Node Classification | **TAGrader** | `Cora`, `PubMed`, `ogbn-products`, `Arxiv-2023` | 🎯 Accuracy | 🧾 Node Classification on TAGs |
| 🧩 Node Classification | **SEGA** | `DAIC-WOZ`, `EATD` | 📊 Macro-F1 | 💬 Depression Detection |
| 🧩 Node Classification | **SocioHyperNet** | `MBTI` | 🎯 Accuracy, 🚨 AUC, 📊 Macro-F1, Micro-F1, IMP | 🧠 Personality Traits |
| 🧩 Node Classification | **Cella** | `Cora`, `Citeseer`, `PubMed`, `Wiki-CS` | 🎯 Accuracy, 🔗 NMI, 📊 ARI, F1 | 🏷️ Label-free Node Classification |
| 🧩 Node Classification | **LLM-TIKG** | `threat-dataset` | ✅ Precision, 🔍 Recall, 📊 F1 | 🛡️ Threat Intelligence KG Construction |
| 🧩 Node Classification | **ANLM-assInNNER** | `NE dataset` | ✅ Precision, 🔍 Recall, 📊 F1 | 🤖 Robotic Fault Diagnosis KG Construction |
| 🧩 Node Classification | **LLM-HetGDT** | `Twitter-HetDrug` | 📊 Macro-F1, ⚖️ GMean | 💊 Online Drug Trafficking Detection |

---

#### 🔹 Class Imbalance Graph Learning – Prediction
| Tasks | Method | Datasets | Metrics | Downstream |
|-------|--------|----------|---------|-------------|
| 🔮 Prediction | **LLM-SBCL** | `biology`, `law`, `cardiff20102`, `sydney19351`, `sydney23146` | 🎯 Accuracy, 📊 Binary-F1, Micro-F1, Macro-F1 | 🎓 Student Performance Prediction |
| 🔮 Prediction | **LKPNR** | `MIND` | 🚨 AUC, 📊 MRR, nDCG | 📰 Personalized News Recommendation |
| 🔮 Prediction | **LLM-DDA** | `BCFR-dataset` | 🚨 AUC, 📊 AUPR, F1, ✅ Precision | 💊 Computational Drug Repositioning |

---

#### 🔹 Class Imbalance Graph Learning – Graph Completion
| Tasks | Method | Datasets | Metrics | Downstream |
|-------|--------|----------|---------|-------------|
| 🔗 Graph Completion | **KICGPT** | `FB15k-237`, `WN18RR` | 📊 MRR, Hits@N | 🔗 Link Completion |
| 🔗 Graph Completion | **KGCD** | `WN18RR`, `YAGO3-10`, `WN18` | 📊 MRR, Hits@N | 🌱 Low-resource KGC |

---

#### 🔹 Class Imbalance Graph Learning – Foundation Model
| Tasks | Method | Datasets | Metrics | Downstream |
|-------|--------|----------|---------|-------------|
| 🏛️ Foundation Model | **GraphCLIP** | `ogbn-arXiv`, `Arxiv-2023`, `PubMed`, `ogbn-products`, `Reddit`, `Cora`, `CiteSeer`, `Ele-Photo`, `Ele-Computers`, `Books-History`, `WikiCS`, `Instagram` | 🎯 Accuracy | 🔄 Transfer Learning on TAGs |

---

#### 🔹 Structure Imbalance Graph Learning
| Tasks | Method | Datasets | Metrics | Downstream |
|-------|--------|----------|---------|-------------|
| 🧩 Node Classification | **GraphEdit** | `Cora`, `Citeseer`, `PubMed` | 🎯 Accuracy | ✏️ Refining Graph Topologies |
| 🔗 Graph Completion | **SATKGC** | `WN18RR`, `FB15k-237`, `Wikidata5M` | 📊 MRR, Hits@N | 🔗 Knowledge Graph Completion |
| 🔗 Graph Completion | **MPIKGC** | `FB15k-237`, `WN18RR`, `FB13`, `WN11` | 📊 MR, MRR, Hits@N, 🎯 Accuracy | 🔗 Knowledge Graph Completion |
| 🔗 Graph Completion | **LLM4RGNN** | `Cora`, `Citeseer`, `PubMed`, `ogbn-arxiv`, `ogbn-products` | 🎯 Accuracy | 🛡️ Improving Adversarial Robustness |

---

## Cross-Domain Heterogeneity in Graphs

> Graphs with heterogeneous node/edge types, multimodal attributes, or domain-specific patterns require specialized methods.

###  Text-Attributed Graph Learning
<!--
| Title | Authors | Venue & Year | Link |
|-------|---------|---------------|------|
| Hypergraph Foundation Model | Yifan Feng, Shiquan Liu, Xiangmin Han, Shaoyi Du, Zongze Wu, Han Hu, Yue Gao | arXiv preprint, 2025 | [Link](https://arxiv.org/abs/2503.01203) |
| UniGraph: Learning a Cross-Domain Graph Foundation Model From Natural Language | Yufei He, Bryan Hooi | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2402.13630) |
| LLM-Align: Utilizing Large Language Models for Entity Alignment in Knowledge Graphs | Xuan Chen, Tong Lu, Zhichun Wang | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2412.04690) |
| Bootstrapping Heterogeneous Graph Representation Learning via Large Language Models: A Generalized Approach | Hang Gao, Chenhao Zhang, Fengge Wu, Junsuo Zhao, Changwen Zheng, Huaping Liu | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2412.08038) |
| Harnessing Explanations: LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning | Xiaoxin He, Xavier Bresson, Thomas Laurent, Adam Perold, Yann LeCun, Bryan Hooi | International Conference on Learning Representations, 2024 | [Link](https://openreview.net/forum?id=RXFVcynVe1) |
| LLMRec: Large Language Models with Graph Augmentation for Recommendation | Wei Wei, Xubin Ren, Jiabin Tang, Qinyong Wang, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, Chao Huang | Proceedings of the 17th ACM International Conference on Web Search and Data Mining, 2024 | [Link](https://doi.org/10.1145/3616855.3635853) |
| Multimodal Fusion of EHR in Structures and Semantics: Integrating Clinical Records and Notes with Hypergraph and LLM | Hejie Cui, Xinyu Fang, Ran Xu, Xuan Kan, Joyce C. Ho, Carl Yang | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2403.08818) |
| One for All: Towards Training One Graph Model for All Classification Tasks | Hao Liu, Jiarui Feng, Lecheng Kong, Ningyue Liang, Dacheng Tao, Yixin Chen, Muhan Zhang | International Conference on Learning Representations, 2024 | [Link](https://openreview.net/forum?id=4IT2pgc9v6) |
-->
<details>
<summary><u><strong>Hypergraph Foundation Model (2025)</strong></u></summary>

**Authors:** Yifan Feng, Shiquan Liu, Xiangmin Han, Shaoyi Du, Zongze Wu, Han Hu, Yue Gao  
**Venue & Year:** arXiv preprint, 2025  
**Link:** [https://arxiv.org/abs/2503.01203](https://arxiv.org/abs/2503.01203)
</details>
<details>
<summary><u><strong>UniGraph: Learning a Cross-Domain Graph Foundation Model From Natural Language (2024)</strong></u></summary>

**Authors:** Yufei He, Bryan Hooi  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2402.13630](https://arxiv.org/abs/2402.13630)
</details>
<details>
<summary><u><strong>LLM-Align: Utilizing Large Language Models for Entity Alignment in Knowledge Graphs (2024)</strong></u></summary>

**Authors:** Xuan Chen, Tong Lu, Zhichun Wang  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2412.04690](https://arxiv.org/abs/2412.04690)
</details>
<details>
<summary><u><strong>Bootstrapping Heterogeneous Graph Representation Learning via Large Language Models: A Generalized Approach (2024)</strong></u></summary>

**Authors:** Hang Gao, Chenhao Zhang, Fengge Wu, Junsuo Zhao, Changwen Zheng, Huaping Liu  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2412.08038](https://arxiv.org/abs/2412.08038)
</details>
<details>
<summary><u><strong>Harnessing Explanations: LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning (2024)</strong></u></summary>

**Authors:** Xiaoxin He, Xavier Bresson, Thomas Laurent, Adam Perold, Yann LeCun, Bryan Hooi  
**Venue & Year:** International Conference on Learning Representations, 2024  
**Link:** [https://openreview.net/forum?id=RXFVcynVe1](https://openreview.net/forum?id=RXFVcynVe1)
</details>
<details>
<summary><u><strong>LLMRec: Large Language Models with Graph Augmentation for Recommendation (2024)</strong></u></summary>

**Authors:** Wei Wei, Xubin Ren, Jiabin Tang, Qinyong Wang, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, Chao Huang  
**Venue & Year:** Proceedings of the 17th ACM International Conference on Web Search and Data Mining, 2024  
**Link:** [https://doi.org/10.1145/3616855.3635853](https://doi.org/10.1145/3616855.3635853)
</details>
<details>
<summary><u><strong>Multimodal Fusion of EHR in Structures and Semantics: Integrating Clinical Records and Notes with Hypergraph and LLM (2024)</strong></u></summary>

**Authors:** Hejie Cui, Xinyu Fang, Ran Xu, Xuan Kan, Joyce C. Ho, Carl Yang  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2403.08818](https://arxiv.org/abs/2403.08818)
</details>
<details>
<summary><u><strong>One for All: Towards Training One Graph Model for All Classification Tasks (2024)</strong></u></summary>

**Authors:** Hao Liu, Jiarui Feng, Lecheng Kong, Ningyue Liang, Dacheng Tao, Yixin Chen, Muhan Zhang  
**Venue & Year:** International Conference on Learning Representations, 2024  
**Link:** [https://openreview.net/forum?id=4IT2pgc9v6](https://openreview.net/forum?id=4IT2pgc9v6)
</details>

---

###  Multimodal Attributed Graph Learning
<!--
| Title | Authors | Venue & Year | Link |
|-------|---------|---------------|------|
| UniGraph2: Learning a Unified Embedding Space to Bind Multimodal Graphs | Yufei He, Yuan Sui, Xiaoxin He, Yue Liu, Yifei Sun, Bryan Hooi | arXiv preprint, 2025 | [Link](https://arxiv.org/abs/2502.00806) |
| LLMRec: Large language models with graph augmentation for recommendation | Wei Wei, Xubin Ren, Jiabin Tang, Qinyong Wang, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, Chao Huang | Proceedings of the 17th ACM International Conference on Web Search and Data Mining, 2024 | [Link](https://doi.org/10.1145/3616855.3635853) |
| Touchup-G: Improving feature representation through graph-centric finetuning | Jing Zhu, Xiang Song, Vassilis Ioannidis, Danai Koutra, Christos Faloutsos | Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2024 | [Link](https://doi.org/10.1145/3626772.3657978) |
| GraphAdapter: Tuning vision-language models with dual knowledge graph | Xin Li, Dongze Lian, Zhihe Lu, Jiawang Bai, Zhibo Chen, Xinchao Wang | Advances in Neural Information Processing Systems, 2024 | [Link](https://openreview.net/forum?id=YmEDnMynuO&noteId=0rFYtJNqHc) |
| When Graph meets Multimodal: Benchmarking and Meditating on Multimodal Attributed Graphs Learning | Hao Yan, Chaozhuo Li, Jun Yin, Zhigang Yu, Weihao Han, Mingzheng Li, Zhengxin Zeng, Hao Sun, Senzhang Wang | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2410.09132) |
| Multimodal Graph Learning for Generative Tasks | Minji Yoon, Jing Yu Koh, Bryan Hooi, Russ Salakhutdinov | NeurIPS 2023 Workshop: New Frontiers in Graph Learning, 2023 | [Link](https://openreview.net/forum?id=YILik4gFBk) |
-->
<details>
<summary><u><strong>UniGraph2: Learning a Unified Embedding Space to Bind Multimodal Graphs (2025)</strong></u></summary>

**Authors:** Yufei He, Yuan Sui, Xiaoxin He, Yue Liu, Yifei Sun, Bryan Hooi  
**Venue & Year:** arXiv preprint, 2025  
**Link:** [https://arxiv.org/abs/2502.00806](https://arxiv.org/abs/2502.00806)
</details>
<details>
<summary><u><strong>LLMRec: Large language models with graph augmentation for recommendation (2024)</strong></u></summary>

**Authors:** Wei Wei, Xubin Ren, Jiabin Tang, Qinyong Wang, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, Chao Huang  
**Venue & Year:** Proceedings of the 17th ACM International Conference on Web Search and Data Mining, 2024  
**Link:** [https://doi.org/10.1145/3616855.3635853](https://doi.org/10.1145/3616855.3635853)
</details>
<details>
<summary><u><strong>Touchup-G: Improving feature representation through graph-centric finetuning (2024)</strong></u></summary>

**Authors:** Jing Zhu, Xiang Song, Vassilis Ioannidis, Danai Koutra, Christos Faloutsos  
**Venue & Year:** Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2024  
**Link:** [https://doi.org/10.1145/3626772.3657978](https://doi.org/10.1145/3626772.3657978)
</details>
<details>
<summary><u><strong>GraphAdapter: Tuning vision-language models with dual knowledge graph (2024)</strong></u></summary>

**Authors:** Xin Li, Dongze Lian, Zhihe Lu, Jiawang Bai, Zhibo Chen, Xinchao Wang  
**Venue & Year:** Advances in Neural Information Processing Systems, 2024  
**Link:** [https://openreview.net/forum?id=YmEDnMynuO&noteId=0rFYtJNqHc](https://openreview.net/forum?id=YmEDnMynuO&noteId=0rFYtJNqHc)
</details>
<details>
<summary><u><strong>When Graph meets Multimodal: Benchmarking and Meditating on Multimodal Attributed Graphs Learning (2024)</strong></u></summary>

**Authors:** Hao Yan, Chaozhuo Li, Jun Yin, Zhigang Yu, Weihao Han, Mingzheng Li, Zhengxin Zeng, Hao Sun, Senzhang Wang  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2410.09132](https://arxiv.org/abs/2410.09132)
</details>
<details>
<summary><u><strong>Multimodal Graph Learning for Generative Tasks (2023)</strong></u></summary>

**Authors:** Minji Yoon, Jing Yu Koh, Bryan Hooi, Russ Salakhutdinov  
**Venue & Year:** NeurIPS 2023 Workshop: New Frontiers in Graph Learning, 2023  
**Link:** [https://openreview.net/forum?id=YILik4gFBk](https://openreview.net/forum?id=YILik4gFBk)
</details>

---


###  Structural Heterogeneous Graph Learning

<!--
| Title | Authors | Venue & Year | Link |
|-------|---------|---------------|------|
| Beyond Graphs: Can Large Language Models Comprehend Hypergraphs? | Yifan Feng, Chengwu Yang, Xingliang Hou, Shaoyi Du, Shihui Ying, Zongze Wu, Yue Gao | International Conference on Learning Representations, 2025 | [Link](https://openreview.net/forum?id=28qOQwjuma) |
| UniGraph2: Learning a Unified Embedding Space to Bind Multimodal Graphs | Yufei He, Yuan Sui, Xiaoxin He, Yue Liu, Yifei Sun, Bryan Hooi | arXiv preprint, 2025 | [Link](https://arxiv.org/abs/2502.00806) |
| Path-LLM: A Shortest-Path-based LLM Learning for Unified Graph Representation | Wenbo Shang, Xuliang Zhu, Xin Huang | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2408.05456) |
| Graphadapter: Tuning vision-language models with dual knowledge graph | Xin Li, Dongze Lian, Zhihe Lu, Jiawang Bai, Zhibo Chen, Xinchao Wang | Advances in Neural Information Processing Systems, 2024 | [Link](https://openreview.net/forum?id=YmEDnMynuO&noteId=0rFYtJNqHc) |
| LLMRec: Large language models with graph augmentation for recommendation | Wei Wei, Xubin Ren, Jiabin Tang, Qinyong Wang, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, Chao Huang | Proceedings of the 17th ACM International Conference on Web Search and Data Mining, 2024 | [Link](https://doi.org/10.1145/3616855.3635853) |
| Talk like a Graph: Encoding Graphs for Large Language Models | Bahare Fatemi, Jonathan Halcrow, Bryan Perozzi | International Conference on Learning Representations, 2024 | [Link](https://openreview.net/forum?id=IuXR1CCrSi) |
| Gita: Graph to visual and textual integration for vision-language graph reasoning | Yanbin Wei, Shuai Fu, Weisen Jiang, Zejian Zhang, Zhixiong Zeng, Qi Wu, James Kwok, Yu Zhang | Advances in Neural Information Processing Systems, 2024 | [Link](https://openreview.net/forum?id=SaodQ13jga) |
| WalkLM: A uniform language model fine-tuning framework for attributed graph embedding | Yanchao Tan, Zihao Zhou, Hang Lv, Weiming Liu, Carl Yang | Advances in Neural Information Processing Systems, 2024 | [Link](https://openreview.net/forum?id=ZrG8kTbt70) |
| Language is all a graph needs | Ruosong Ye, Caiqi Zhang, Runhui Wang, Shuyuan Xu, Yongfeng Zhang | Findings of the Association for Computational Linguistics: EACL, 2024 | [Link](https://aclanthology.org/2024.findings-eacl.132/) |
| Graph neural prompting with large language models | Yijun Tian, Huan Song, Zichen Wang, Haozhu Wang, Ziqing Hu, Fang Wang, Nitesh V Chawla, Panpan Xu | Proceedings of the AAAI Conference on Artificial Intelligence, 2024 | [Link](https://doi.org/10.1609/aaai.v38i17.29875) |
| Let your graph do the talking: Encoding structured data for LLMs | Bryan Perozzi, Bahare Fatemi, Dustin Zelle, Anton Tsitsulin, Mehran Kazemi, Rami Al-Rfou, Jonathan Halcrow | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2402.05862) |
| GraphGPT: Graph instruction tuning for large language models | Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Suqi Cheng, Dawei Yin, Chao Huang | Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2024 | [Link](https://doi.org/10.1145/3626772.3657775) |
| Higpt: Heterogeneous graph language model | Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Long Xia, Dawei Yin, Chao Huang | Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024 | [Link](https://doi.org/10.1145/3637528.3671987) |
| GPT4Graph: Can large language models understand graph structured data? An empirical evaluation and benchmarking | Jiayan Guo, Lun Du, Hengyu Liu, Mengyu Zhou, Xinyi He, Shi Han | arXiv preprint, 2023 | [Link](https://arxiv.org/abs/2305.15066) |
| Graphtext: Graph reasoning in text space | Jianan Zhao, Le Zhuo, Yikang Shen, Meng Qu, Kai Liu, Michael Bronstein, Zhaocheng Zhu, Jian Tang | arXiv preprint, 2023 | [Link](https://arxiv.org/abs/2310.01089) |
| Can language models solve graph problems in natural language? | Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, Yulia Tsvetkov | Advances in Neural Information Processing Systems, 2023 | [Link](https://openreview.net/forum?id=UDqHhbqYJV) |
| Evaluating large language models on graphs: Performance insights and comparative analysis | Chang Liu, Bo Wu | arXiv preprint, 2023 | [Link](https://arxiv.org/abs/2308.11224) |
-->
<details>
<summary><u><strong>Beyond Graphs: Can Large Language Models Comprehend Hypergraphs? (2025)</strong></u></summary>

**Authors:** Yifan Feng, Chengwu Yang, Xingliang Hou, Shaoyi Du, Shihui Ying, Zongze Wu, Yue Gao  
**Venue & Year:** International Conference on Learning Representations, 2025  
**Link:** [https://openreview.net/forum?id=28qOQwjuma](https://openreview.net/forum?id=28qOQwjuma)
</details>
<details>
<summary><u><strong>UniGraph2: Learning a Unified Embedding Space to Bind Multimodal Graphs (2025)</strong></u></summary>

**Authors:** Yufei He, Yuan Sui, Xiaoxin He, Yue Liu, Yifei Sun, Bryan Hooi  
**Venue & Year:** arXiv preprint, 2025  
**Link:** [https://arxiv.org/abs/2502.00806](https://arxiv.org/abs/2502.00806)
</details>
<details>
<summary><u><strong>Path-LLM: A Shortest-Path-based LLM Learning for Unified Graph Representation (2024)</strong></u></summary>

**Authors:** Wenbo Shang, Xuliang Zhu, Xin Huang  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2408.05456](https://arxiv.org/abs/2408.05456)
</details>
<details>
<summary><u><strong>Graphadapter: Tuning vision-language models with dual knowledge graph (2024)</strong></u></summary>

**Authors:** Xin Li, Dongze Lian, Zhihe Lu, Jiawang Bai, Zhibo Chen, Xinchao Wang  
**Venue & Year:** Advances in Neural Information Processing Systems, 2024  
**Link:** [https://openreview.net/forum?id=YmEDnMynuO&noteId=0rFYtJNqHc](https://openreview.net/forum?id=YmEDnMynuO&noteId=0rFYtJNqHc)
</details>
<details>
<summary><u><strong>LLMRec: Large language models with graph augmentation for recommendation (2024)</strong></u></summary>

**Authors:** Wei Wei, Xubin Ren, Jiabin Tang, Qinyong Wang, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, Chao Huang  
**Venue & Year:** Proceedings of the 17th ACM International Conference on Web Search and Data Mining, 2024  
**Link:** [https://doi.org/10.1145/3616855.3635853](https://doi.org/10.1145/3616855.3635853)
</details>
<details>
<summary><u><strong>Talk like a Graph: Encoding Graphs for Large Language Models (2024)</strong></u></summary>

**Authors:** Bahare Fatemi, Jonathan Halcrow, Bryan Perozzi  
**Venue & Year:** International Conference on Learning Representations, 2024  
**Link:** [https://openreview.net/forum?id=IuXR1CCrSi](https://openreview.net/forum?id=IuXR1CCrSi)
</details>
<details>
<summary><u><strong>Gita: Graph to visual and textual integration for vision-language graph reasoning (2024)</strong></u></summary>

**Authors:** Yanbin Wei, Shuai Fu, Weisen Jiang, Zejian Zhang, Zhixiong Zeng, Qi Wu, James Kwok, Yu Zhang  
**Venue & Year:** Advances in Neural Information Processing Systems, 2024  
**Link:** [https://openreview.net/forum?id=SaodQ13jga](https://openreview.net/forum?id=SaodQ13jga)
</details>
<details>
<summary><u><strong>WalkLM: A uniform language model fine-tuning framework for attributed graph embedding (2024)</strong></u></summary>

**Authors:** Yanchao Tan, Zihao Zhou, Hang Lv, Weiming Liu, Carl Yang  
**Venue & Year:** Advances in Neural Information Processing Systems, 2024  
**Link:** [https://openreview.net/forum?id=ZrG8kTbt70](https://openreview.net/forum?id=ZrG8kTbt70)
</details>
<details>
<summary><u><strong>Language is all a graph needs (2024)</strong></u></summary>

**Authors:** Ruosong Ye, Caiqi Zhang, Runhui Wang, Shuyuan Xu, Yongfeng Zhang  
**Venue & Year:** Findings of the Association for Computational Linguistics: EACL, 2024  
**Link:** [https://aclanthology.org/2024.findings-eacl.132/](https://aclanthology.org/2024.findings-eacl.132/)
</details>
<details>
<summary><u><strong>Graph neural prompting with large language models (2024)</strong></u></summary>

**Authors:** Yijun Tian, Huan Song, Zichen Wang, Haozhu Wang, Ziqing Hu, Fang Wang, Nitesh V Chawla, Panpan Xu  
**Venue & Year:** Proceedings of the AAAI Conference on Artificial Intelligence, 2024  
**Link:** [https://doi.org/10.1609/aaai.v38i17.29875](https://doi.org/10.1609/aaai.v38i17.29875)
</details>
<details>
<summary><u><strong>Let your graph do the talking: Encoding structured data for LLMs (2024)</strong></u></summary>

**Authors:** Bryan Perozzi, Bahare Fatemi, Dustin Zelle, Anton Tsitsulin, Mehran Kazemi, Rami Al-Rfou, Jonathan Halcrow  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2402.05862](https://arxiv.org/abs/2402.05862)
</details>
<details>
<summary><u><strong>GraphGPT: Graph instruction tuning for large language models (2024)</strong></u></summary>

**Authors:** Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Suqi Cheng, Dawei Yin, Chao Huang  
**Venue & Year:** Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2024  
**Link:** [https://doi.org/10.1145/3626772.3657775](https://doi.org/10.1145/3626772.3657775)
</details>
<details>
<summary><u><strong>Higpt: Heterogeneous graph language model (2024)</strong></u></summary>

**Authors:** Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Long Xia, Dawei Yin, Chao Huang  
**Venue & Year:** Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024  
**Link:** [https://doi.org/10.1145/3637528.3671987](https://doi.org/10.1145/3637528.3671987)
</details>
<details>
<summary><u><strong>GPT4Graph: Can large language models understand graph structured data? An empirical evaluation and benchmarking (2023)</strong></u></summary>

**Authors:** Jiayan Guo, Lun Du, Hengyu Liu, Mengyu Zhou, Xinyi He, Shi Han  
**Venue & Year:** arXiv preprint, 2023  
**Link:** [https://arxiv.org/abs/2305.15066](https://arxiv.org/abs/2305.15066)
</details>
<details>
<summary><u><strong>Graphtext: Graph reasoning in text space (2023)</strong></u></summary>

**Authors:** Jianan Zhao, Le Zhuo, Yikang Shen, Meng Qu, Kai Liu, Michael Bronstein, Zhaocheng Zhu, Jian Tang  
**Venue & Year:** arXiv preprint, 2023  
**Link:** [https://arxiv.org/abs/2310.01089](https://arxiv.org/abs/2310.01089)
</details>
<details>
<summary><u><strong>Can language models solve graph problems in natural language? (2023)</strong></u></summary>

**Authors:** Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, Yulia Tsvetkov  
**Venue & Year:** Advances in Neural Information Processing Systems, 2023  
**Link:** [https://openreview.net/forum?id=UDqHhbqYJV](https://openreview.net/forum?id=UDqHhbqYJV)
</details>
<details>
<summary><u><strong>Evaluating large language models on graphs: Performance insights and comparative analysis (2023)</strong></u></summary>

**Authors:** Chang Liu, Bo Wu  
**Venue & Year:** arXiv preprint, 2023  
**Link:** [https://arxiv.org/abs/2308.11224](https://arxiv.org/abs/2308.11224)
</details>

---

### Datasets, Metrics, and Tasks (Cross-domain Heterogeneity)

#### 🔹 Text-Attributed Graph Learning – Textual Attribute Alignment
| Tasks | Method | Datasets | Metrics | Downstream |
|-------|--------|----------|---------|-------------|
| 📝 Textual Alignment | **TAPE** | `Cora`, `PubMed`, `Arxiv-2023`, `ogbn-arxiv`, `ogbn-products` | 🎯 Accuracy | 🧩 Node Classification |
| 📝 Textual Alignment | **LLMRec** | `MovieLens`, `Netflix` | ✅ Recall, nDCG, Precision | 🎬 Item Recommendation |
| 📝 Textual Alignment | **MINGLE** | `MIMIC-III`, `CRADLE` | 🎯 Accuracy, 🚨 AUC, 📊 AUPR, F1 | 🧩 Node Classification |
| 📝 Textual Alignment | **GHGRL** | `IMDB`, `DBLP`, `ACM`, `Wiki-CS`, `IMDB-RIR`, `DBLP-RID` | 📊 Macro-F1, Micro-F1 | 🧩 Node Classification |

#### 🔹 Text-Attributed Graph Learning – Graph Foundation Models
| Tasks | Method | Datasets | Metrics | Downstream |
|-------|--------|----------|---------|-------------|
| 🏛️ GFM | **OFA** | `Cora`, `PubMed`, `ogbn-arxiv`, `Wiki-CS`, `MOLHIV`, `MOLPCBA`, `FB15K237`, `WN18RR`, `ChEMBL` | 🎯 Accuracy, 🚨 AUC, 📊 AUPR | 🧩 Node / 🔗 Link / 📦 Graph Classification |
| 🏛️ GFM | **UniGraph** | `Cora`, `PubMed`, `ogbn-arxiv`, `ogbn-products`, `Wiki-CS`, `FB15K237`, `WN18RR`, `MOLHIV`, `MOLPCBA` | 🚨 AUC | 🧩 Node / 🔗 Link / 📦 Graph Classification |
| 🏛️ GFM | **BooG** | `Cora`, `PubMed`, `ogbn-arxiv`, `Wiki-CS`, `MOLHIV`, `MOLPCBA` | 🚨 AUC | 🧩 Node / 📦 Graph Classification |
| 🏛️ GFM | **Hyper-FM** | `Cora-CA-Text`, `Cora-CC-Text`, `Pubmed-CA-Text`, `Pubmed-CC-Text`, `AminerText`, `Arxiv-Text`, `Movielens-Text`, `IMDB-Text`, `GoodBook-Text`, `PPI-Text` | 🎯 Accuracy | 🧩 Node Classification |

---

#### 🔹 Multimodal Attributed Graph Learning – MLLM-based Alignment
| Tasks | Method | Datasets | Metrics | Downstream |
|-------|--------|----------|---------|-------------|
| 🖼️ MLLM Align | **LLMRec** | `MovieLens`, `Netflix` | ✅ Recall, nDCG, Precision | 🎬 Item Recommendation |
| 🖼️ MLLM Align | **MAGB** | `Cora`, `Wiki-CS`, `Ele-Photo`, `Flickr`, `Movies`, `Toys`, `Grocery`, `Reddit-S`, `Reddit-M` | 🎯 Accuracy, 📊 F1 | 🧩 Node Classification |

#### 🔹 Multimodal Attributed Graph Learning – Graph-Enhanced Alignment
| Tasks | Method | Datasets | Metrics | Downstream |
|-------|--------|----------|---------|-------------|
| 🖼️ Graph Align | **MMGL** | `WikiWeb2M` | 📝 BLEU-4, ROUGE-L, CIDEr | 📑 Section Summarization |
| 🖼️ Graph Align | **GraphAdapter** | `ImageNet`, `StanfordCars`, `UCF101`, `Caltech101`, `Flowers102`, `SUN397`, `DTD`, `EuroSAT`, `FGVCAircraft`, `OxfordPets`, `Food101` | 🎯 Accuracy | 🖼️ Image Classification |
| 🖼️ Graph Align | **TouchUp-G** | `ogbn-arxiv`, `ogbn-products`, `Books`, `Amazon-CP` | 📊 MRR, Hits@N, 🎯 Accuracy | 🔗 Link Prediction, 🧩 Node Classification |
| 🖼️ Graph Align | **UniGraph2** | `Cora`, `PubMed`, `ogbn-arxiv`, `ogbn-papers100M`, `ogbn-products`, `Wiki-CS`, `FB15K237`, `WN18RR`, `Amazon-Sports`, `Amazon-Cloth`, `Goodreads-LP`, `Goodreads-NC`, `Ele-Fashion`, `WikiWeb2M` | 🎯 Accuracy, 📝 BLEU-4, ROUGE-L, CIDEr | 🧩 Node / 🧷 Edge Classification / 📑 Section Summarization |

---

#### 🔹 Structural Heterogeneous Graph Learning – Topological Graph Textualization
| Tasks | Method | Datasets | Metrics | Downstream |
|-------|--------|----------|---------|-------------|
| 🌐 Topo Text | **LLMtoGraph** | `Synthetic Graph Data` | 🎯 Accuracy, 🔎 Positive Response Ratio | 🧩 Node Classification, 🔍 Path Finding, 🧮 Pattern Matching |
| 🌐 Topo Text | **NLGraph** | `NLGraph` | 🎯 Accuracy, 📏 Partial Credit Score, Relative Error | 🔍 Path Finding, 🧮 Pattern Matching, 🔄 Topological Sort |
| 🌐 Topo Text | **Talk like a Graph** | `GraphQA` | 🎯 Accuracy | 🔗 Link Prediction, 🧮 Pattern Matching |
| 🌐 Topo Text | **GPT4Graph** | `ogbn-arxiv`, `MOLHIV`, `MOLPCBA`, `MetaQA` | 🎯 Accuracy | 🧩 Node / 📦 Graph Classification, ❓ Graph Query Language |
| 🌐 Topo Text | **GITA** | `GVLQA` | 🎯 Accuracy | 🔗 Link Prediction, 🧮 Pattern Matching, 🔍 Path Finding, 🔄 Topological Sort |
| 🌐 Topo Text | **LLM4-Hypergraph** | `LLM4Hypergraph` | 🎯 Accuracy | 🌀 Isomorphism Recognition, 📦 Structure Classification, 🔗 Link Prediction, 🔍 Path Finding |

---

#### 🔹 Structural Heterogeneous Graph Learning – Attributed Graph Textualization
| Tasks | Method | Datasets | Metrics | Downstream |
|-------|--------|----------|---------|-------------|
| 🧩 Attrib Text | **GraphText** | `Cora`, `Citeseer`, `Texas`, `Wisconsin`, `Cornell` | 🎯 Accuracy | 🧩 Node Classification |
| 🧩 Attrib Text | **WalkLM** | `PubMed`, `MIMIC-III` | 📊 Macro-F1, Micro-F1, 🚨 AUC, MRR | 🧩 Node / 🔗 Link Prediction |
| 🧩 Attrib Text | **Path-LLM** | `Cora`, `Citeseer`, `PubMed`, `ogbn-arxiv` | 📊 Macro-F1, Micro-F1, 🚨 AUC, 🎯 Accuracy | 🧩 Node / 🔗 Link Prediction |
| 🧩 Attrib Text | **InstructGLM** | `Cora`, `PubMed`, `ogbn-arxiv` | 🎯 Accuracy | 🧩 Node / 🔗 Link Prediction |
| 🧩 Attrib Text | **MuseGraph** | `Cora`, `ogbn-arxiv`, `MIMIC-III`, `AGENDA`, `WebNLG` | 📊 Macro-F1, Micro-F1, Weighted-F1, 📝 BLEU-4, METEOR, ROUGE-L, CHRF++ | 🧩 Node Classification, ✍️ Graph-to-Text Generation |
| 🧩 Attrib Text | **Graph-LLM** | `Cora`, `Citeseer`, `PubMed`, `ogbn-arxiv`, `ogbn-products` | 🎯 Accuracy | 🧩 Node Classification |

---

#### 🔹 Structural Heterogeneous Graph Learning – Graph Token Learning
| Tasks | Method | Datasets | Metrics | Downstream |
|-------|--------|----------|---------|-------------|
| 🏷️ Token | **GNP** | `OBQA`, `ARC`, `PIQA`, `Riddle`, `PQA`, `BioASQ` | 🎯 Accuracy | ❓ Question Answering |
| 🏷️ Token | **GraphToken** | `GraphQA` | 🎯 Accuracy | 🔗 Link Prediction, 🧮 Pattern Matching |
| 🏷️ Token | **GraphGPT** | `Cora`, `PubMed`, `ogbn-arxiv` | 🎯 Accuracy, 📊 Macro-F1, 🚨 AUC | 🧩 Node / 🔗 Link Prediction |
| 🏷️ Token | **LLaGA** | `Cora`, `PubMed`, `ogbn-arxiv`, `ogbn-products` | 🎯 Accuracy | 🧩 Node / 🔗 Link Prediction |
| 🏷️ Token | **TEA-GLM** | `Cora`, `PubMed`, `ogbn-arxiv`, `TAG benchmark` | 🎯 Accuracy, 🚨 AUC | 🧩 Node / 🔗 Link Prediction |
| 🏷️ Token | **HiGPT** | `IMDB`, `DBLP`, `ACM` | 📊 Macro-F1, Micro-F1, 🚨 AUC | 🧩 Node Classification |

---

## Dynamic Instability in Graphs

> Graph structures may evolve over time or require adaptive interaction. These works explore LLMs in dynamic graph settings.

###  Querying and Reasoning
<!--
| Title | Authors | Venue & Year | Link |
|-------|---------|---------------|------|
| LLM4DyG: Can Large Language Models Solve Spatial-Temporal Problems on Dynamic Graphs? | Zeyang Zhang, Xin Wang, Ziwei Zhang, Haoyang Li, Yijian Qin, Wenwu Zhu | Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024 | [Link](https://doi.org/10.1145/3637528.3671709) |
| TimeR$^4$: Time-aware Retrieval-Augmented Large Language Models for Temporal Knowledge Graph Question Answering | Xinying Qian, Ying Zhang, Yu Zhao, Baohang Zhou, Xuhui Sui, Li Zhang, Kehui Song | Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Miami, USA | [Link](https://aclanthology.org/2024.emnlp-main.394/) |
| Two-stage Generative Question Answering on Temporal Knowledge Graph Using Large Language Models | Yifu Gao, Linbo Qiao, Zhigang Kan, Zhihua Wen, Yongquan He, Dongsheng Li | Findings of the Association for Computational Linguistics: ACL 2024, Bangkok, Thailand | [Link](https://aclanthology.org/2024.findings-acl.401/) |
| Unveiling LLMs: The Evolution of Latent Representations in a Dynamic Knowledge Graph | Marco Bronzini, Carlo Nicolini, Bruno Lepri, Jacopo Staiano, Andrea Passerini | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2404.03623) |
| Large Language Models Can Learn Temporal Reasoning | Siheng Xiong, Ali Payani, Ramana Kompella, Faramarz Fekri | Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, 2024 | [Link](https://aclanthology.org/2024.acl-long.563/) |
| Chain-of-History Reasoning for Temporal Knowledge Graph Forecasting | Yuwei Xia, Ding Wang, Qiang Liu, Liang Wang, Shu Wu, Xiao-Yu Zhang | Findings of the Association for Computational Linguistics: ACL 2024, Bangkok, Thailand | [Link](https://aclanthology.org/2024.findings-acl.955/) |
| zrLLM: Zero-Shot Relational Learning on Temporal Knowledge Graphs with Large Language Models | Zifeng Ding, Heling Cai, Jingpei Wu, Yunpu Ma, Ruotong Liao, Bo Xiong, Volker Tresp | Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Mexico City, Mexico | [Link](https://aclanthology.org/2024.naacl-long.104/) |
| Temporal Knowledge Graph Forecasting Without Knowledge Using In-Context Learning | Dong-Ho Lee, Kian Ahrabian, Woojeong Jin, Fred Morstatter, Jay Pujara | Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Singapore | [Link](https://aclanthology.org/2023.emnlp-main.36/) |
-->
## Robust Graph Learning – Temporal & Dynamic Graphs

<details>
<summary><u><strong>LLM4DyG: Can Large Language Models Solve Spatial-Temporal Problems on Dynamic Graphs? (2024)</strong></u></summary>

**Authors:** Zeyang Zhang, Xin Wang, Ziwei Zhang, Haoyang Li, Yijian Qin, Wenwu Zhu  
**Venue & Year:** Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024  
**Link:** [https://doi.org/10.1145/3637528.3671709](https://doi.org/10.1145/3637528.3671709)
</details>
<details>
<summary><u><strong>TimeR$^4$: Time-aware Retrieval-Augmented Large Language Models for Temporal Knowledge Graph Question Answering (2024)</strong></u></summary>

**Authors:** Xinying Qian, Ying Zhang, Yu Zhao, Baohang Zhou, Xuhui Sui, Li Zhang, Kehui Song  
**Venue & Year:** Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Miami, USA  
**Link:** [https://aclanthology.org/2024.emnlp-main.394/](https://aclanthology.org/2024.emnlp-main.394/)
</details>
<details>
<summary><u><strong>Two-stage Generative Question Answering on Temporal Knowledge Graph Using Large Language Models (2024)</strong></u></summary>

**Authors:** Yifu Gao, Linbo Qiao, Zhigang Kan, Zhihua Wen, Yongquan He, Dongsheng Li  
**Venue & Year:** Findings of the Association for Computational Linguistics: ACL 2024, Bangkok, Thailand  
**Link:** [https://aclanthology.org/2024.findings-acl.401/](https://aclanthology.org/2024.findings-acl.401/)
</details>
<details>
<summary><u><strong>Unveiling LLMs: The Evolution of Latent Representations in a Dynamic Knowledge Graph (2024)</strong></u></summary>

**Authors:** Marco Bronzini, Carlo Nicolini, Bruno Lepri, Jacopo Staiano, Andrea Passerini  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2404.03623](https://arxiv.org/abs/2404.03623)
</details>
<details>
<summary><u><strong>Large Language Models Can Learn Temporal Reasoning (2024)</strong></u></summary>

**Authors:** Siheng Xiong, Ali Payani, Ramana Kompella, Faramarz Fekri  
**Venue & Year:** Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, 2024  
**Link:** [https://aclanthology.org/2024.acl-long.563/](https://aclanthology.org/2024.acl-long.563/)
</details>
<details>
<summary><u><strong>Chain-of-History Reasoning for Temporal Knowledge Graph Forecasting (2024)</strong></u></summary>

**Authors:** Yuwei Xia, Ding Wang, Qiang Liu, Liang Wang, Shu Wu, Xiao-Yu Zhang  
**Venue & Year:** Findings of the Association for Computational Linguistics: ACL 2024, Bangkok, Thailand  
**Link:** [https://aclanthology.org/2024.findings-acl.955/](https://aclanthology.org/2024.findings-acl.955/)
</details>
<details>
<summary><u><strong>zrLLM: Zero-Shot Relational Learning on Temporal Knowledge Graphs with Large Language Models (2024)</strong></u></summary>

**Authors:** Zifeng Ding, Heling Cai, Jingpei Wu, Yunpu Ma, Ruotong Liao, Bo Xiong, Volker Tresp  
**Venue & Year:** Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Mexico City, Mexico  
**Link:** [https://aclanthology.org/2024.naacl-long.104/](https://aclanthology.org/2024.naacl-long.104/)
</details>
<details>
<summary><u><strong>Temporal Knowledge Graph Forecasting Without Knowledge Using In-Context Learning (2023)</strong></u></summary>

**Authors:** Dong-Ho Lee, Kian Ahrabian, Woojeong Jin, Fred Morstatter, Jay Pujara  
**Venue & Year:** Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Singapore  
**Link:** [https://aclanthology.org/2023.emnlp-main.36/](https://aclanthology.org/2023.emnlp-main.36/)
</details>

---

###  Generating and Updating
<!--
| Title | Authors | Venue & Year | Link |
|-------|---------|---------------|------|
| FinDKG: Dynamic Knowledge Graphs with Large Language Models for Detecting Global Trends in Financial Markets | Xiaohui Victor Li, Francesco Sanna Passino | ACM, 2024 | [Link](https://doi.org/10.1145/3677052.3698603) |
| GenTKG: Generative Forecasting on Temporal Knowledge Graph with Large Language Models | Ruotong Liao, Xu Jia, Yangzhe Li, Yunpu Ma, Volker Tresp | Findings of the Association for Computational Linguistics: NAACL, 2024 | [Link](https://aclanthology.org/2024.findings-naacl.268/) |
| Up To Date: Automatic Updating Knowledge Graphs Using LLMs | Shahenda Hatem, Ghada Khoriba, Mohamed H. Gad-Elrab, Mohamed ElHelw | Procedia Computer Science, 2024 | [Link](https://www.sciencedirect.com/science/article/pii/S1877050924030072) |
| Pre-trained Language Model with Prompts for Temporal Knowledge Graph Completion | Wenjie Xu, Ben Liu, Miao Peng, Xu Jia, Min Peng | Findings of the Association for Computational Linguistics: ACL, 2023 | [Link](https://aclanthology.org/2023.findings-acl.493/) |
| Large Language Models-guided Dynamic Adaptation for Temporal Knowledge Graph Reasoning | Jiapu Wang, Kai Sun, Linhao Luo, Wei Wei, Yongli Hu, Alan Wee-Chung Liew, Shirui Pan, Baocai Yin | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2405.14170) |
| Back to the Future: Towards Explainable Temporal Reasoning with Large Language Models | Chenhan Yuan, Qianqian Xie, Jimin Huang, Sophia Ananiadou | ACM, 2024 | [Link](https://doi.org/10.1145/3589334.3645376) |
| RealTCD: Temporal Causal Discovery from Interventional Data with Large Language Model | Peiwen Li, Xin Wang, Zeyang Zhang, Yuan Meng, Fang Shen, Yue Li, Jialong Wang, Yang Li, Wenwu Zhu | ACM, 2024 | [Link](https://doi.org/10.1145/3627673.3680042) |
| DynLLM: When Large Language Models Meet Dynamic Graph Recommendation | Ziwei Zhao, Fake Lin, Xi Zhu, Zhi Zheng, Tong Xu, Shitian Shen, Xueying Li, Zikai Yin, Enhong Chen | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2405.07580) |
-->
<details>
<summary><u><strong>FinDKG: Dynamic Knowledge Graphs with Large Language Models for Detecting Global Trends in Financial Markets (2024)</strong></u></summary>

**Authors:** Xiaohui Victor Li, Francesco Sanna Passino  
**Venue & Year:** ACM, 2024  
**Link:** [https://doi.org/10.1145/3677052.3698603](https://doi.org/10.1145/3677052.3698603)
</details>
<details>
<summary><u><strong>GenTKG: Generative Forecasting on Temporal Knowledge Graph with Large Language Models (2024)</strong></u></summary>

**Authors:** Ruotong Liao, Xu Jia, Yangzhe Li, Yunpu Ma, Volker Tresp  
**Venue & Year:** Findings of the Association for Computational Linguistics: NAACL, 2024  
**Link:** [https://aclanthology.org/2024.findings-naacl.268/](https://aclanthology.org/2024.findings-naacl.268/)
</details>
<details>
<summary><u><strong>Up To Date: Automatic Updating Knowledge Graphs Using LLMs (2024)</strong></u></summary>

**Authors:** Shahenda Hatem, Ghada Khoriba, Mohamed H. Gad-Elrab, Mohamed ElHelw  
**Venue & Year:** Procedia Computer Science, 2024  
**Link:** [https://www.sciencedirect.com/science/article/pii/S1877050924030072](https://www.sciencedirect.com/science/article/pii/S1877050924030072)
</details>
<details>
<summary><u><strong>Pre-trained Language Model with Prompts for Temporal Knowledge Graph Completion (2023)</strong></u></summary>

**Authors:** Wenjie Xu, Ben Liu, Miao Peng, Xu Jia, Min Peng  
**Venue & Year:** Findings of the Association for Computational Linguistics: ACL, 2023  
**Link:** [https://aclanthology.org/2023.findings-acl.493/](https://aclanthology.org/2023.findings-acl.493/)
</details>
<details>
<summary><u><strong>Large Language Models-guided Dynamic Adaptation for Temporal Knowledge Graph Reasoning (2024)</strong></u></summary>

**Authors:** Jiapu Wang, Kai Sun, Linhao Luo, Wei Wei, Yongli Hu, Alan Wee-Chung Liew, Shirui Pan, Baocai Yin  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2405.14170](https://arxiv.org/abs/2405.14170)
</details>
<details>
<summary><u><strong>Back to the Future: Towards Explainable Temporal Reasoning with Large Language Models (2024)</strong></u></summary>

**Authors:** Chenhan Yuan, Qianqian Xie, Jimin Huang, Sophia Ananiadou  
**Venue & Year:** ACM, 2024  
**Link:** [https://doi.org/10.1145/3589334.3645376](https://doi.org/10.1145/3589334.3645376)
</details>
<details>
<summary><u><strong>RealTCD: Temporal Causal Discovery from Interventional Data with Large Language Model (2024)</strong></u></summary>

**Authors:** Peiwen Li, Xin Wang, Zeyang Zhang, Yuan Meng, Fang Shen, Yue Li, Jialong Wang, Yang Li, Wenwu Zhu  
**Venue & Year:** ACM, 2024  
**Link:** [https://doi.org/10.1145/3627673.3680042](https://doi.org/10.1145/3627673.3680042)
</details>
<details>
<summary><u><strong>DynLLM: When Large Language Models Meet Dynamic Graph Recommendation (2024)</strong></u></summary>

**Authors:** Ziwei Zhao, Fake Lin, Xi Zhu, Zhi Zheng, Tong Xu, Shitian Shen, Xueying Li, Zikai Yin, Enhong Chen  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2405.07580](https://arxiv.org/abs/2405.07580)
</details>

---

###  Evaluation and Application
<!--
| Title | Authors | Venue & Year | Link |
|-------|---------|---------------|------|
| DARG: Dynamic Evaluation of Large Language Models via Adaptive Reasoning Graph | Zhehao Zhang, Jiaao Chen, Diyi Yang | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2406.17271) |
| AnomalyLLM: Few-shot Anomaly Edge Detection for Dynamic Graphs using Large Language Models | Shuo Liu, Di Yao, Lanting Fang, Zhetao Li, Wenbin Li, Kaiyu Feng, XiaoWen Ji, Jingping Bi | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2405.07626) |
| Language-Grounded Dynamic Scene Graphs for Interactive Object Search With Mobile Manipulation | Daniel Honerkamp, Martin Büchner, Fabien Despinoy, Tim Welschehold, Abhinav Valada | IEEE Robotics and Automation Letters, 2024 | [Link](https://doi.org/10.1109/LRA.2024.3441495) |
| Temporal Relational Reasoning of Large Language Models for Detecting Stock Portfolio Crashes | Kelvin J. L. Koa, Yunshan Ma, Ritchie Ng, Huanhuan Zheng, Tat-Seng Chua | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2410.17266) |
| Dynamic Benchmarking of Masked Language Models on Temporal Concept Drift with Multiple Views | Katerina Margatina, Shuai Wang, Yogarshi Vyas, Neha Anna John, Yassine Benajiba, Miguel Ballesteros | Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, 2023 | [Link](https://aclanthology.org/2023.eacl-main.211/) |
-->

## Robust Graph Learning – Dynamic Evaluation & Temporal Reasoning

<details>
<summary><u><strong>DARG: Dynamic Evaluation of Large Language Models via Adaptive Reasoning Graph (2024)</strong></u></summary>

**Authors:** Zhehao Zhang, Jiaao Chen, Diyi Yang  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2406.17271](https://arxiv.org/abs/2406.17271)
</details>
<details>
<summary><u><strong>AnomalyLLM: Few-shot Anomaly Edge Detection for Dynamic Graphs using Large Language Models (2024)</strong></u></summary>

**Authors:** Shuo Liu, Di Yao, Lanting Fang, Zhetao Li, Wenbin Li, Kaiyu Feng, XiaoWen Ji, Jingping Bi  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2405.07626](https://arxiv.org/abs/2405.07626)
</details>
<details>
<summary><u><strong>Language-Grounded Dynamic Scene Graphs for Interactive Object Search With Mobile Manipulation (2024)</strong></u></summary>

**Authors:** Daniel Honerkamp, Martin Büchner, Fabien Despinoy, Tim Welschehold, Abhinav Valada  
**Venue & Year:** IEEE Robotics and Automation Letters, 2024  
**Link:** [https://doi.org/10.1109/LRA.2024.3441495](https://doi.org/10.1109/LRA.2024.3441495)
</details>
<details>
<summary><u><strong>Temporal Relational Reasoning of Large Language Models for Detecting Stock Portfolio Crashes (2024)</strong></u></summary>

**Authors:** Kelvin J. L. Koa, Yunshan Ma, Ritchie Ng, Huanhuan Zheng, Tat-Seng Chua  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2410.17266](https://arxiv.org/abs/2410.17266)
</details>
<details>
<summary><u><strong>Dynamic Benchmarking of Masked Language Models on Temporal Concept Drift with Multiple Views (2023)</strong></u></summary>

**Authors:** Katerina Margatina, Shuai Wang, Yogarshi Vyas, Neha Anna John, Yassine Benajiba, Miguel Ballesteros  
**Venue & Year:** Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, 2023  
**Link:** [https://aclanthology.org/2023.eacl-main.211/](https://aclanthology.org/2023.eacl-main.211/)
</details>
---

### Datasets, Metrics, and Tasks (Dynamic Instability)

#### 🔹 Querying & Reasoning — Forecasting & Reasoning
| Tasks | Method | Datasets | Metrics | Downstream |
|------|--------|----------|---------|------------|
| 🔮 Forecasting / Reasoning | **ICL** | `WIKI`, `YAGO`, `ICEWS14`, `ICEWS18` | 📊 MRR, Hits@N | 🔗 Link Prediction |
| 🔮 Forecasting / Reasoning | **zrLLM** | `ICEWS`, `ACLED` | 📊 MRR, Hits@N | 🔗 Link Prediction |
| 🔮 Forecasting / Reasoning | **CoH** | `ICEWS14`, `ICEWS18`, `ICEWS05-15` | 📊 MRR, Hits@N | 🔗 Link Prediction |
| 🔮 Forecasting / Reasoning | **TG-LLM** | `TGQA`, `TimeQA`, `TempReason` | 🎯 F1, Accuracy, ✅ Exact Match | ⏱️ Temporal Reasoning |
| 🔮 Forecasting / Reasoning | **LLM4DyG** | `Enron`, `DBLP`, `Flights` | 🎯 Accuracy, 📊 F1, 🔍 Recall | 🗺️ Spatio-Temporal & Graph Reasoning / Querying, 🔗 Link Prediction |

#### 🔹 Querying & Reasoning — QA & Interpretability
| Tasks | Method | Datasets | Metrics | Downstream |
|------|--------|----------|---------|------------|
| ❓ Temporal KGQA | **TimeR<sup>4</sup>** | `MULTITQ`, `TimeQuestions` | 📊 Hits@N | ❓ Temporal KGQA |
| ❓ Temporal KGQA | **GenTKGQA** | `CronQuestion`, `TimeQuestions` | 📊 Hits@N | ❓ Temporal KGQA |
| 🔍 Interpretability / Claim Verification | **Unveiling LLMs** | `FEVER`, `CLIMATE-FEVER` | ✅ Precision, 🔍 Recall, 📊 F1, 🧪 ROC AUC, 🎯 Accuracy | 🧾 Claim Verification |

---

#### 🧬 Generating & Updating — Generating Structures
| Tasks | Method | Datasets | Metrics | Downstream |
|------|--------|----------|---------|------------|
| 🧱 Structure Generation | **FinDKG** | `WIKI`, `YAGO`, `ICEWS14` | 📊 MRR, Hits@N | 🔗 Link Prediction |
| 🧱 Structure Generation | **GenTKG** | `ICEWS14`, `ICEWS18`, `GDELT`, `YAGO` | 📊 Hits@N | 🔗 Link Prediction |
| 🧱 Structure Generation | **Up To Date** | `Wikidata` | 🎯 Accuracy, 📈 Response Rate | ✅ Fact Validation, ❓ QA |
| 🧱 Structure Generation | **PPT** | `ICEWS14`, `ICEWS18`, `ICEWS05-15` | 📊 MRR, Hits@N | 🔗 Link Prediction |
| 🧱 Structure Generation | **LLM-DA** | `ICEWS14`, `ICEWS05-15` | 📊 MRR, Hits@N | 🔗 Link Prediction |

#### 🧬 Generating & Updating — Generating Insights & Representations
| Tasks | Method | Datasets | Metrics | Downstream |
|------|--------|----------|---------|------------|
| 💡 Insights / Explanations | **TimeLlama** | `ICEWS14`, `ICEWS18`, `ICEWS05-15` | ✅ Precision, 🔍 Recall, 📊 F1, 📝 BLEU, ROUGE | 📅 Event Forecasting, 🗒️ Explanation Generation |
| 💡 Causal / Anomaly | **RealTCD** | `Simulation Datasets` | 🧭 SHD, SID | 🧠 Temporal Causal Discovery, ⚠️ Anomaly Detection |
| 💡 Dynamic RecSys | **DynLLM** | `Tmall`, `Alibaba` | 🔁 Recall@K, nDCG@K | 🎯 Dynamic Graph Recommendation / Top-K Recommendation |

---

#### 🧪 Evaluation & Application — Model Evaluation
| Tasks | Method | Datasets | Metrics | Downstream |
|------|--------|----------|---------|------------|
| 🧷 Temporal Robustness / Probing | **Dynamic-TempLAMA** | `DYNAMICTEMPLAMA` | 🎯 Accuracy, 📊 MRR, 📝 ROUGE, 📊 F1 | 🕒 Temporal Robustness Evaluation, 📚 Factual Knowledge Probing |
| 🧮 Dynamic Reasoning Eval | **DARG** | `GSM8K`, `BBQ`, `BBH Navigate`, `BBH Dyck Language` | 🎯 Accuracy, 📐 CIAR, ✅ Exact Match, 🎯 Accuracy | ➗ Mathematical / 👥 Social / 🧭 Spatial / 🔣 Symbolic Reasoning |

#### 🧪 Evaluation & Application — Downstream Applications
| Tasks | Method | Datasets | Metrics | Downstream |
|------|--------|----------|---------|------------|
| ⚠️ Anomaly Detection | **AnomalyLLM** | `UCI Messages`, `Blogcatalog` | 🚨 AUC | ⚠️ Anomaly Detection |
| 🤖 Interactive Object Search | **MoMa-LLM** | `iGibson scenes` | 🚨 AUC, 🔍 Recall | 🗺️ Semantic Interactive Object Search |
| 📰 Event Detection (Finance) | **TRR** | `Reuters Financial News` | 🧪 AUROC | 📰 Event Detection |

---




## 📖 Citation

If you find our work useful, please consider citing the following paper:
```bibtex
@article{li2025survey,
  title={A Survey of Large Language Models for Data Challenges in Graphs},
  author={Li, Mengran and Zhang, Pengyu and Xing, Wenbin and Zheng, Yijia and Zaporojets, Klim and Chen, Junzhou and Zhang, Ronghui and Zhang, Yong and Gong, Siyuan and Hu, Jia and others},
  journal={Expert Systems with Applications},
  pages={129643},
  year={2025},
  publisher={Elsevier}
}

@article{li2025using,
  title={Using Large Language Models to Tackle Fundamental Challenges in Graph Learning: A Comprehensive Survey},
  author={Li, Mengran and Zhang, Pengyu and Xing, Wenbin and Zheng, Yijia and Zaporojets, Klim and Chen, Junzhou and Zhang, Ronghui and Zhang, Yong and Gong, Siyuan and Hu, Jia and Ma, Xiaolei and Liu, Zhiyuan and Groth, Paul and Worring, Marcel},
  journal={arXiv preprint arXiv:2505.18475},
  year={2025}
}
