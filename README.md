
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
<summary><u><strong> Spatiotemporal Pre-Trained Large Language Model for Forecasting With Missing Values (2025)</strong></u></summary>

**Authors:** Le Fang, Wei Xiang, Shirui Pan, Flora D. Salim, Yi-Ping Phoebe Chen  
**Venue & Year:** IEEE Internet of Things Journal, 2025  
**Link:** [https://doi.org/10.1109/JIOT.2024.3524030](https://doi.org/10.1109/JIOT.2024.3524030)  
**Abstract:**  
Spatiotemporal data collected by sensors within an urban Internet of Things (IoT) system inevitably contains some missing values, which significantly affects the accuracy of spatiotemporal data forecasting. However, existing techniques, including those based on large language models (LLMs), show limited effectiveness in forecasting with missing values, especially in scenarios involving high-dimensional sensor data. In this article, we propose a novel spatiotemporal pretrained LLM dubbed SPLLM for forecasting with missing values. In this network, we seamlessly integrate a specialized spatiotemporal fusion graph convolutional network (GCN) module that extracts intricate spatiotemporal and graph-based information, for generating suitable inputs to the SPLLM. Furthermore, we propose a feed-forward network (FFN) fine-tuning strategy within the LLM and a final fusion layer to enable the model to leverage the pretrained foundational knowledge of the LLM and adapt to new incomplete data simultaneously. The experimental results indicate that SPLLM outperforms state-of-the-art models on real-world public datasets. Notably, SPLLM exhibits a superior performance in tackling incomplete sensory data with a variety of missing rates. A comprehensive ablation study of key components is conducted to demonstrate their efficiency.
</details>
<details>
<summary><u><strong> LLM as Prompter: Low-resource Inductive Reasoning on Arbitrary Knowledge Graphs (2024)</strong></u></summary>

**Authors:** Kai Wang, Yuwei Xu, Zhiyong Wu, Siqiang Luo  
**Venue & Year:** Findings of the Association for Computational Linguistics, 2024  
**Link:** [https://doi.org/10.18653/v1/2024.findings-acl.224](https://doi.org/10.18653/v1/2024.findings-acl.224)  
**Abstract:**  
Knowledge Graph (KG) inductive reasoning, which aims to infer missing facts from new KGs that are not seen during training, has been widely adopted in various applications. One critical challenge of KG inductive reasoning is handling low-resource scenarios with scarcity in both textual and structural aspects. In this paper, we attempt to address this challenge with Large Language Models (LLMs). Particularly, we utilize the state-of-the-art LLMs to generate a graph-structural prompt to enhance the pre-trained Graph Neural Networks (GNNs), which brings us new methodological insights into the KG inductive reasoning methods, as well as high generalizability in practice. On the methodological side, we introduce a novel pretraining and prompting framework ProLINK, designed for low-resource inductive reasoning across arbitrary KGs without requiring additional training. On the practical side, we experimentally evaluate our approach on 36 low-resource KG datasets and find that ProLINK outperforms previous methods in three-shot, one-shot, and zero-shot reasoning tasks, exhibiting average performance improvements by 20%, 45%, and 147%, respectively. Furthermore, ProLINK demonstrates strong robustness for various LLM promptings as well as full-shot scenarios.

</details>
<details>
<summary><u><strong> On LLM-Enhanced Mixed-Type Data Imputation with High-Order Message Passing (2025)</strong></u></summary>

**Authors:** Jianwei Wang, Kai Wang, Ying Zhang, Wenjie Zhang, Xiwei Xu, Xuemin Lin  
**Venue & Year:** arXiv preprint, 2025  
**Link:** [https://arxiv.org/abs/2501.02191](https://arxiv.org/abs/2501.02191)  
**Abstract:**  
Missing data imputation, which aims to impute the missing values in the raw datasets to achieve the completeness of datasets, is crucial for modern data-driven models like large language models (LLMs) and has attracted increasing interest over the past decades. Despite its importance, existing solutions for missing data imputation either 1) only support numerical and categorical data or 2) show an unsatisfactory performance due to their design prioritizing text data and the lack of key properties for tabular data imputation. In this paper, we propose UnIMP, a Unified IMPutation framework that leverages LLM and high-order message passing to enhance the imputation of mixed-type data including numerical, categorical, and text data. Specifically, we first introduce a cell-oriented hypergraph to model the table. We then propose BiHMP, an efficient Bidirectional High-order Message-Passing network to aggregate global-local information and high-order relationships on the constructed hypergraph while capturing the inter-column heterogeneity and intra-column homogeneity. To effectively and efficiently align the capacity of the LLM with the information aggregated by BiHMP, we introduce Xfusion, which, together with BiHMP, acts as adapters for the LLM. We follow a pre-training and fine-tuning pipeline to train UnIMP, integrating two optimizations: chunking technique, which divides tables into smaller chunks to enhance efficiency; and progressive masking technique, which gradually adapts the model to learn more complex data patterns. Both theoretical proofs and empirical experiments on 10 real world datasets highlight the superiority of UnIMP over existing techniques.
</details>
<details>
<summary><u><strong> Large language models as topological structure enhancers for text-attributed graphs (2023)</strong></u></summary>

**Authors:** Shengyin Sun, Yuxiang Ren, Chen Ma, Xuecang Zhang  
**Venue & Year:** arXiv preprint, 2023  
**Link:** [https://doi.org/10.48550/arXiv.2311.14324](https://doi.org/10.48550/arXiv.2311.14324)  
**Abstract:**  
The latest advancements in large language models (LLMs) have revolutionized the field of natural language processing (NLP). Inspired by the success of LLMs in NLP tasks, some recent work has begun investigating the potential of applying LLMs in graph learning tasks. However, most of the existing work focuses on utilizing LLMs as powerful node feature augmenters, leaving employing LLMs to enhance graph topological structures an understudied problem. In this work, we explore how to leverage the information retrieval and text generation capabilities of LLMs to refine/enhance the topological structure of text-attributed graphs (TAGs) under the node classification setting. First, we propose using LLMs to help remove unreliable edges and add reliable ones in the TAG. Specifically, we first let the LLM output the semantic similarity between node attributes through delicate prompt designs, and then perform edge deletion and edge addition based on the similarity. Second, we propose using pseudo-labels generated by the LLM to improve graph topology, that is, we introduce the pseudo-label propagation as a regularization to guide the graph neural network (GNN) in learning proper edge weights. Finally, we incorporate the two aforementioned LLM-based methods for graph topological refinement into the process of GNN training, and perform extensive experiments on four real-world datasets. The experimental results demonstrate the effectiveness of LLM-based graph topology refinement (achieving a 0.15%--2.47% performance gain on public benchmarks).
</details>
<details>
<summary><u><strong> Leveraging Large Language Models for Node Generation in Few-Shot Learning on Text-Attributed Graphs (2025)</strong></u></summary>

**Authors:** Jianxiang Yu, Yuxiang Ren, Chenghua Gong, Jiaqi Tan, Xiang Li, Xuecang Zhang  
**Venue & Year:** AAAI, 2025  
**Link:** [https://doi.org/10.48550/arXiv.2310.09872](https://doi.org/10.48550/arXiv.2310.09872)  
**Abstract:**  
Text-attributed graphs have recently garnered significant attention due to their wide range of applications in web domains. Existing methodologies employ word embedding models for acquiring text representations as node features, which are subsequently fed into Graph Neural Networks (GNNs) for training. Recently, the advent of Large Language Models (LLMs) has introduced their powerful capabilities in information retrieval and text generation, which can greatly enhance the text attributes of graph data. Furthermore, the acquisition and labeling of extensive datasets are both costly and time-consuming endeavors. Consequently, few-shot learning has emerged as a crucial problem in the context of graph learning tasks. In order to tackle this challenge, we propose a lightweight paradigm called LLM4NG, which adopts a plug-and-play approach to empower text-attributed graphs through node generation using LLMs. Specifically, we utilize LLMs to extract semantic information from the labels and generate samples that belong to these categories as exemplars. Subsequently, we employ an edge predictor to capture the structural information inherent in the raw dataset and integrate the newly generated samples into the original graph. This approach harnesses LLMs for enhancing class-level information and seamlessly introduces labeled nodes and edges without modifying the raw dataset, thereby facilitating the node classification task in few-shot scenarios. Extensive experiments demonstrate the outstanding performance of our proposed paradigm, particularly in low-shot scenarios. For instance, in the 1-shot setting of the ogbn-arxiv dataset, LLM4NG achieves a 76% improvement over the baseline model.
</details>
<details>
<summary><u><strong> Label-free node classification on graphs with large language models (LLMs) (2023)</strong></u></summary>

**Authors:** Zhikai Chen, Haitao Mao, Hongzhi Wen, Haoyu Han, Wei Jin, Haiyang Zhang, Hui Liu, Jiliang Tang  
**Venue & Year:** arXiv preprint, 2023  
**Link:** [https://arxiv.org/abs/2310.04668](https://arxiv.org/abs/2310.04668)  
**Abstract:**  
In recent years, there have been remarkable advancements in node classification achieved by Graph Neural Networks (GNNs). However, they necessitate abundant high-quality labels to ensure promising performance. In contrast, Large Language Models (LLMs) exhibit impressive zero-shot proficiency on text-attributed graphs. Yet, they face challenges in efficiently processing structural data and suffer from high inference costs. In light of these observations, this work introduces a label-free node classification on graphs with LLMs pipeline, LLM-GNN. It amalgamates the strengths of both GNNs and LLMs while mitigating their limitations. Specifically, LLMs are leveraged to annotate a small portion of nodes and then GNNs are trained on LLMs' annotations to make predictions for the remaining large portion of nodes. The implementation of LLM-GNN faces a unique challenge: how can we actively select nodes for LLMs to annotate and consequently enhance the GNN training? How can we leverage LLMs to obtain annotations of high quality, representativeness, and diversity, thereby enhancing GNN performance with less cost? To tackle this challenge, we develop an annotation quality heuristic and leverage the confidence scores derived from LLMs to advanced node selection. Comprehensive experimental results validate the effectiveness of LLM-GNN. In particular, LLM-GNN can achieve an accuracy of 74.9% on a vast-scale dataset \products with a cost less than 1 dollar.
</details>
<details>
<summary><u><strong> GraphLLM: Boosting graph reasoning ability of large language models (2023)</strong></u></summary>

**Authors:** Ziwei Chai, Tianjie Zhang, Liang Wu, Kaiqiao Han, Xiaohai Hu, Xuanwen Huang, Yang Yang  
**Venue & Year:** arXiv preprint, 2023  
**Link:** [https://doi.org/10.48550/arXiv.2310.05845](https://doi.org/10.48550/arXiv.2310.05845)  
**Abstract:**  
The advancement of Large Language Models (LLMs) has remarkably pushed the boundaries towards artificial general intelligence (AGI), with their exceptional ability on understanding diverse types of information, including but not limited to images and audio. Despite this progress, a critical gap remains in empowering LLMs to proficiently understand and reason on graph data. Recent studies underscore LLMs' underwhelming performance on fundamental graph reasoning tasks. In this paper, we endeavor to unearth the obstacles that impede LLMs in graph reasoning, pinpointing the common practice of converting graphs into natural language descriptions (Graph2Text) as a fundamental bottleneck. To overcome this impediment, we introduce GraphLLM, a pioneering end-to-end approach that synergistically integrates graph learning models with LLMs. This synergy equips LLMs with the ability to proficiently interpret and reason on graph data, harnessing the superior expressive power of graph learning models. Our empirical evaluations across four fundamental graph reasoning tasks validate the effectiveness of GraphLLM. The results exhibit a substantial average accuracy enhancement of 54.44%, alongside a noteworthy context reduction of 96.45% across various graph reasoning tasks.
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
**Abstract:**  
Knowledge Graph Question Generation (KGQG) is the task of generating natural language questions based on the given knowledge graph (KG). Although extensively explored in recent years, prevailing models predominantly depend on labelled data for training deep learning models or employ large parametric frameworks, e.g., Large Language Models (LLMs), which can incur significant deployment costs and pose practical implementation challenges. To address these issues, in this work, we put forward a zero-shot, multi-agent KGQG framework. This framework integrates the capabilities of LLMs with small models to facilitate cost-effective, high-quality question generation. In specific, we develop a professional editorial team architecture accompanied by two workflow optimization tools to reduce unproductive collaboration among LLMs-based agents and enhance the robustness of the system. Extensive experiments demonstrate that our proposed framework derives the new state-of-the-art performance on the zero-shot KGQG tasks, with relative gains of 20.24% and 13.57% on two KGQG datasets, respectively, which rival fully supervised state-of-the-art models.
</details>
<details>
<summary><u><strong>LinkGPT: Teaching Large Language Models to Predict Missing Links (2024)</strong></u></summary>

**Authors:** Zhongmou He, Jing Zhu, Shengyi Qian, Joyce Chai, Danai Koutra  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://doi.org/10.48550/arXiv.2406.04640](https://doi.org/10.48550/arXiv.2406.04640)  
**Abstract:**  
</details>
<details>
<summary><u><strong>AnomalyLLM: Few-shot Anomaly Edge Detection for Dynamic Graphs using Large Language Models (2024)</strong></u></summary>

**Authors:** Shuo Liu, Di Yao, Lanting Fang, Zhetao Li, Wenbin Li, Kaiyu Feng, XiaoWen Ji, Jingping Bi  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2405.07626](https://arxiv.org/abs/2405.07626)  
**Abstract:**  
Large Language Models (LLMs) have shown promising results on various language and vision tasks. Recently, there has been growing interest in applying LLMs to graph-based tasks, particularly on Text-Attributed Graphs (TAGs). However, most studies have focused on node classification, while the use of LLMs for link prediction (LP) remains understudied. In this work, we propose a new task on LLMs, where the objective is to leverage LLMs to predict missing links between nodes in a graph. This task evaluates an LLM's ability to reason over structured data and infer new facts based on learned patterns. This new task poses two key challenges: (1) How to effectively integrate pairwise structural information into the LLMs, which is known to be crucial for LP performance, and (2) how to solve the computational bottleneck when teaching LLMs to perform LP. To address these challenges, we propose LinkGPT, the first end-to-end trained LLM for LP tasks. To effectively enhance the LLM's ability to understand the underlying structure, we design a two-stage instruction tuning approach where the first stage fine-tunes the pairwise encoder, projector, and node projector, and the second stage further fine-tunes the LLMs to predict links. To address the efficiency challenges at inference time, we introduce a retrieval-reranking scheme. Experiments show that LinkGPT can achieve state-of-the-art performance on real-world graphs as well as superior generalization in zero-shot and few-shot learning, surpassing existing benchmarks. At inference time, it can achieve 10\times speedup while maintaining high LP accuracy.
</details>
<details>
<summary><u><strong>HeGTa: Leveraging Heterogeneous Graph-enhanced Large Language Models for Few-shot Complex Table Understanding (2024)</strong></u></summary>

**Authors:** Rihui Jin, Yu Li, Guilin Qi, Nan Hu, Yuan-Fang Li, Jiaoyan Chen, Jianan Wang, Yongrui Chen, Dehai Min, Sheng Bi  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2403.19723](https://arxiv.org/abs/2403.19723)  
**Abstract:**  
Table understanding (TU) has achieved promising advancements, but it faces the challenges of the scarcity of manually labeled tables and the presence of complex table this http URL address these challenges, we propose HGT, a framework with a heterogeneous graph (HG)-enhanced large language model (LLM) to tackle few-shot TU this http URL leverages the LLM by aligning the table semantics with the LLM's parametric knowledge through soft prompts and instruction turning and deals with complex tables by a multi-task pre-training scheme involving three novel multi-granularity self-supervised HG pre-training this http URL empirically demonstrate the effectiveness of HGT, showing that it outperforms the SOTA for few-shot complex TU on several benchmarks.
</details>
<details>
<summary><u><strong>FlexKBQA: A Flexible LLM-powered Framework for Few-shot Knowledge Base Question Answering (2024)</strong></u></summary>

**Authors:** Zhenyu Li, Sunqi Fan, Yu Gu, Xiuxing Li, Zhichao Duan, Bowen Dong, Ning Liu, Jianyong Wang  
**Venue & Year:** Proceedings of the AAAI Conference on Artificial Intelligence, 2024  
**Link:** [https://doi.org/10.1609/aaai.v38i17.29823](https://doi.org/10.1609/aaai.v38i17.29823)  
**Abstract:**  
Knowledge base question answering (KBQA) is a critical yet challenging task due to the vast number of entities within knowledge bases and the diversity of natural language questions posed by users. Unfortunately, the performance of most KBQA models tends to decline significantly in real-world scenarios where high-quality annotated data is insufficient. To mitigate the burden associated with manual annotation, we introduce FlexKBQA by utilizing Large Language Models (LLMs) as program translators for addressing the challenges inherent in the few-shot KBQA task. Specifically, FlexKBQA leverages automated algorithms to sample diverse programs, such as SPARQL queries, from the knowledge base, which are subsequently converted into natural language questions via LLMs. This synthetic dataset facilitates training a specialized lightweight model for the KB. Additionally, to reduce the barriers of distribution shift between synthetic data and real user questions, FlexKBQA introduces an executionguided self-training method to iterative leverage unlabeled user questions. Furthermore, we explore harnessing the inherent reasoning capability of LLMs to enhance the entire framework. Consequently, FlexKBQA delivers substantial flexibility, encompassing data annotation, deployment, and being domain agnostic. Through extensive experiments on GrailQA, WebQSP, and KQA Pro, we observe that under the few-shot even the more challenging zero-shot scenarios, FlexKBQA achieves impressive results with a few annotations, surpassing all previous baselines and even approaching the performance of supervised models, achieving a remarkable 93% performance relative to the fully-supervised models. We posit that FlexKBQA represents a significant advancement towards exploring better integration of large and lightweight models. Code is available at https://github.com/leezythu/FlexKBQA.
</details>
<details>
<summary><u><strong>Zero-shot Knowledge Graph Question Generation via Multi-agent LLMs and Small Models Synthesis (2024)</strong></u></summary>

**Authors:** Runhao Zhao, Jiuyang Tang, Weixin Zeng, Ziyang Chen, Xiang Zhao  
**Venue & Year:** Proceedings of the ACM International Conference on Information and Knowledge Management, 2024  
**Link:** [https://doi.org/10.1145/3627673.3679805](https://doi.org/10.1145/3627673.3679805)  
**Abstract:**  
Knowledge Graph Question Generation (KGQG) is the task of generating natural language questions based on the given knowledge graph (KG). Although extensively explored in recent years, prevailing models predominantly depend on labelled data for training deep learning models or employ large parametric frameworks, e.g., Large Language Models (LLMs), which can incur significant deployment costs and pose practical implementation challenges. To address these issues, in this work, we put forward a zero-shot, multi-agent KGQG framework. This framework integrates the capabilities of LLMs with small models to facilitate cost-effective, high-quality question generation. In specific, we develop a professional editorial team architecture accompanied by two workflow optimization tools to reduce unproductive collaboration among LLMs-based agents and enhance the robustness of the system. Extensive experiments demonstrate that our proposed framework derives the new state-of-the-art performance on the zero-shot KGQG tasks, with relative gains of 20.24% and 13.57% on two KGQG datasets, respectively, which rival fully supervised state-of-the-art models.
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
**Abstract:**  
Multiple-input multiple-output (MIMO) radar offers several performance and flexibility advantages over traditional radar arrays. However, high angular and Doppler resolutions necessitate a large number of antenna elements and the transmission of numerous chirps, leading to increased hardware and computational complexity. While compressive sensing (CS) has recently been applied to pulsed-waveform radars with sparse measurements, its application to frequency-modulated continuous wave (FMCW) radar for target detection remains largely unexplored. In this paper, we propose a novel CS-based multi-target localization algorithm in the range, Doppler, and angular domains for MIMO-FMCW radar, where we jointly estimate targets' velocities and angles of arrival. To this end, we present a signal model for sparse-random and uniform linear arrays based on three-dimensional spectral estimation. For range estimation, we propose a discrete Fourier transform (DFT)-based focusing and orthogonal matching pursuit (OMP)-based techniques, each with distinct advantages, while two-dimensional CS is used for joint Doppler-angle estimation. Leveraging the properties of structured random matrices, we establish theoretical uniform and non-uniform recovery guarantees with high probability for the proposed framework. Our numerical experiments demonstrate that our methods achieve similar detection performance and higher resolution compared to conventional DFT and MUSIC with fewer transmitted chirps and antenna elements.
</details>
<details>
<summary><u><strong>GS-KGC: A generative subgraph-based framework for knowledge graph completion with large language models (2025)</strong></u></summary>

**Authors:** Rui Yang, Jiahao Zhu, Jianping Man, Hongze Liu, Li Fang, Yi Zhou  
**Venue & Year:** Information Fusion, 2025  
**Link:** [https://doi.org/10.1016/j.inffus.2024.102868](https://doi.org/10.1016/j.inffus.2024.102868)  
**Abstract:**  
Knowledge graph completion (KGC) focuses on identifying missing triples in a knowledge graph (KG) , which is crucial for many downstream applications. Given the rapid development of large language models (LLMs), some LLM-based methods are proposed for KGC task. However, most of them focus on prompt engineering while overlooking the fact that finer-grained subgraph information can aid LLMs in generating more accurate answers. In this paper, we propose a novel completion framework called Generative Subgraph-based KGC (GS-KGC), which utilizes subgraph information as contextual reasoning and employs a QA approach to achieve the KGC task. This framework primarily includes a subgraph partitioning algorithm designed to generate negatives and neighbors. Specifically, negatives can encourage LLMs to generate a broader range of answers, while neighbors provide additional contextual insights for LLM reasoning. Furthermore, we found that GS-KGC can discover potential triples within the KGs and new facts beyond the KGs. Experiments conducted on four common KGC datasets highlight the advantages of the proposed GS-KGC, e.g., it shows a 5.6% increase in Hits@3 compared to the LLM-based model CP-KGC on the FB15k-237N, and a 9.3% increase over the LLM-based model TECHS on the ICEWS14.

</details>
<details>
<summary><u><strong>In-Context Learning with Topological Information for LLM-Based Knowledge Graph Completion (2024)</strong></u></summary>

**Authors:** Udari Madhushani Sehwag, Kassiani Papasotiriou, Jared Vann, Sumitra Ganesh  
**Venue & Year:** ICML 2024 Workshop on Structured Probabilistic Inference & Generative Modeling  
**Link:** [https://openreview.net/forum?id=eUpH8AuVQa](https://openreview.net/forum?id=eUpH8AuVQa)  
**Abstract:**  
Knowledge graphs (KGs) are crucial for representing and reasoning over structured information, supporting a wide range of applications such as information retrieval, question answering, and decision-making. However, their effectiveness is often hindered by incompleteness, limiting their potential for real-world impact. While knowledge graph completion (KGC) has been extensively studied in the literature, recent advances in generative AI models, particularly large language models (LLMs), have introduced new opportunities for innovation. In-context learning has recently emerged as a promising approach for leveraging pretrained knowledge of LLMs across a range of natural language processing tasks and has been widely adopted in both academia and industry. However, how to utilize in-context learning for effective KGC remains relatively underexplored. We develop a novel method that incorporates topological information through in-context learning to enhance KGC performance. By integrating ontological knowledge and graph structure into the context of LLMs, our approach achieves strong performance in the transductive setting i.e., nodes in the test graph dataset are present in the training graph dataset. Furthermore, we apply our approach to KGC in the more challenging inductive setting, i.e., nodes in the training graph dataset and test graph dataset are disjoint, leveraging the ontology to infer useful information about missing nodes which serve as contextual cues for the LLM during inference. Our method demonstrates superior performance compared to baselines on the ILPC-small and ILPC-large datasets.

</details>
<details>
<summary><u><strong>Making large language models perform better in knowledge graph completion (2024)</strong></u></summary>

**Authors:** Yichi Zhang, Zhuo Chen, Lingbing Guo, Yajing Xu, Wen Zhang, Hlmruajun Chen  
**Venue & Year:** Proceedings of the 32nd ACM International Conference on Multimedia, 2024  
**Link:** [https://doi.org/10.1145/3664647.3681327](https://doi.org/10.1145/3664647.3681327)  
**Abstract:**  
Large language model (LLM) based knowledge graph completion (KGC) aims to predict the missing triples in the KGs with LLMs. However, research about LLM-based KGC fails to sufficiently harness LLMs' inference proficiencies, overlooking critical structural information integral to KGs. In this paper, we explore methods to incorporate structural information into the LLMs, with the overarching goal of facilitating structure-aware reasoning. We first discuss on the existing LLM paradigms like in-context learning and instruction tuning, proposing basic structural information injection approaches. Then we propose a Knowledge Prefix Adapter (KoPA) to fulfill this stated goal. KoPA uses a structural pre-training phase to comprehend the intricate entities and relations within KGs, representing them as structural embeddings. Then KoPA communicates such cross-modal structural information understanding to the LLMs through a knowledge prefix adapter which projects the structural embeddings into the textual space and obtains virtual knowledge tokens positioned as a prefix of the input prompt. We conduct comprehensive experiments and provide incisive analysis. Our code and data are available at https://github.com/zjukg/KoPA.
</details>
<details>
<summary><u><strong>LLM-based multi-level knowledge generation for few-shot knowledge graph completion (2024)</strong></u></summary>

**Authors:** Qian Li, Zhuo Chen, Cheng Ji, Shiqi Jiang, Jianxin Li  
**Venue & Year:** Proceedings of the International Joint Conference on Artificial Intelligence, 2024  
**Link:** [https://www.ijcai.org/proceedings/2024/236](https://www.ijcai.org/proceedings/2024/236)  
**Abstract:**  
Knowledge Graphs (KGs) are pivotal in various NLP applications but often grapple with incompleteness, especially due to the long-tail problem where infrequent, unpopular relationships drastically reduce the KG completion performance. In this paper, we focus on Few-shot Knowledge Graph Completion (FKGC), a task addressing these gaps in long-tail scenarios. Amidst the rapid evolution of Large Language Models, we propose a generation-based FKGC paradigm facilitated by LLM distillation. Our MuKDC framework employs multi-level knowledge distillation for few-shot KG completion, generating supplementary knowledge to mitigate data scarcity in few-shot environments. MuKDC comprises two primary components: Multi-level Knowledge Generation, which enriches the KG at various levels, and Consistency Assessment, to ensure the coherence and reliability of the generated knowledge. Most notably, our method achieves SOTA results in both FKGC and multi-modal FKGC benchmarks, significantly advancing KG completion and enhancing the understanding and application of LLMs in structured knowledge generation and assessment.
</details>
<details>
<summary><u><strong>Assessing LLMs Suitability for Knowledge Graph Completion (2024)</strong></u></summary>

**Authors:** Vasile Ionut Remus Iga, Gheorghe Cosmin Silaghi  
**Venue & Year:** International Conference on Neural-Symbolic Learning and Reasoning, 2024  
**Link:** [https://doi.org/10.1007/978-3-031-71170-1_22](https://doi.org/10.1007/978-3-031-71170-1_22)  
**Abstract:**  
Recent work has shown the capability of Large Language Models (LLMs) to solve tasks related to Knowledge Graphs, such as Knowledge Graph Completion, even in Zero- or Few-Shot paradigms. However, they are known to hallucinate answers, or output results in a non-deterministic manner, thus leading to wrongly reasoned responses, even if they satisfy the user’s demands. To highlight opportunities and challenges in knowledge graphs-related tasks, we experiment with three distinguished LLMs, namely Mixtral-8x7b-Instruct-v0.1, GPT-3.5-Turbo-0125 and GPT-4o, on Knowledge Graph Completion for static knowledge graphs, using prompts constructed following the TELeR taxonomy, in Zero- and One-Shot contexts, on a Task-Oriented Dialogue system use case. When evaluated using both strict and flexible metrics measurement manners, our results show that LLMs could be fit for such a task if prompts encapsulate sufficient information and relevant examples.
</details>
<details>
<summary><u><strong>Finetuning generative large language models with discrimination instructions for knowledge graph completion (2024)</strong></u></summary>

**Authors:** Yang Liu, Xiaobin Tian, Zequn Sun, Wei Hu  
**Venue & Year:** International Semantic Web Conference, 2024  
**Link:** [https://doi.org/10.1007/978-3-031-77844-5_11](https://doi.org/10.1007/978-3-031-77844-5_11)  
**Abstract:**  
</details>
<details>
<summary><u><strong>Enhancing text-based knowledge graph completion with zero-shot large language models: A focus on semantic enhancement (2024)</strong></u></summary>

**Authors:** Rui Yang, Jiahao Zhu, Jianping Man, Li Fang, Yi Zhou  
**Venue & Year:** Knowledge-Based Systems, 2024  
**Link:** [https://doi.org/10.1016/j.knosys.2024.112155](https://doi.org/10.1016/j.knosys.2024.112155)  
**Abstract:**  
Traditional knowledge graph (KG) completion models learn embeddings to predict missing facts. Recent works attempt to complete KGs in a text-generation manner with large language models (LLMs). However, they need to ground the output of LLMs to KG entities, which inevitably brings errors. In this paper, we present a finetuning framework, DIFT, aiming to unleash the KG completion ability of LLMs and avoid grounding errors. Given an incomplete fact, DIFT employs a lightweight model to obtain candidate entities and finetunes an LLM with discrimination instructions to select the correct one from the given candidates. To improve performance while reducing instruction data, DIFT uses a truncated sampling method to select useful facts for finetuning and injects KG embeddings into the LLM. Extensive experiments on benchmark datasets demonstrate the effectiveness of our proposed framework.
</details>
<details>
<summary><u><strong>Framing Few-Shot Knowledge Graph Completion with Large Language Models (2023)</strong></u></summary>

**Authors:** Adrian MP Brasoveanu, Lyndon Nixon, Albert Weichselbraun, Arno Scharl  
**Venue & Year:** Joint Workshop Proceedings of the 5th International Workshop on Sem4Tra and SEMANTiCS, 2023  
**Link:** [https://ceur-ws.org/Vol-3510/paper_nlp_4.pdf](https://ceur-ws.org/Vol-3510/paper_nlp_4.pdf)  
**Abstract:**  
Knowledge Graph Completion (KGC) from text involves identifying known or unknown entities (nodes) as well as relations (edges) among these entities. Recent work has started to explore the use of Large Language Models (LLMs) for entity detection and relation extraction, due to their Natural Language Understanding (NLU) capabilities. However, LLM performance varies across models and depends on the quality of the prompt engineering. We examine specific relation extraction cases and present a set of
examples collected from well-known resources in a small corpus. We provide a set of annotations and identify various issues that occur when using different LLMs for this task. As LLMs will remain a focal point of future KGC research, we conclude with suggestions for improving the KGC process.
</details>
<details>
<summary><u><strong>Iterative zero-shot LLM prompting for knowledge graph construction (2023)</strong></u></summary>

**Authors:** Salvatore Carta, Alessandro Giuliani, Leonardo Piano, Alessandro Sebastian Podda, Livio Pompianu, Sandro Gabriele Tiddia  
**Venue & Year:** arXiv preprint, 2023  
**Link:** [https://doi.org/10.48550/ARXIV.2307.01128](https://doi.org/10.48550/ARXIV.2307.01128)  
**Abstract:**  
In the current digitalization era, capturing and effectively representing knowledge is crucial in most real-world scenarios. In this context, knowledge graphs represent a potent tool for retrieving and organizing a vast amount of information in a properly interconnected and interpretable structure. However, their generation is still challenging and often requires considerable human effort and domain expertise, hampering the scalability and flexibility across different application fields. This paper proposes an innovative knowledge graph generation approach that leverages the potential of the latest generative large language models, such as GPT-3.5, that can address all the main critical issues in knowledge graph building. The approach is conveyed in a pipeline that comprises novel iterative zero-shot and external knowledge-agnostic strategies in the main stages of the generation process. Our unique manifold approach may encompass significant benefits to the scientific community. In particular, the main contribution can be summarized by: (i) an innovative strategy for iteratively prompting large language models to extract relevant components of the final graph; (ii) a zero-shot strategy for each prompt, meaning that there is no need for providing examples for "guiding" the prompt result; (iii) a scalable solution, as the adoption of LLMs avoids the need for any external resources or human expertise. To assess the effectiveness of our proposed model, we performed experiments on a dataset that covered a specific domain. We claim that our proposal is a suitable solution for scalable and versatile knowledge graph construction and may be applied to different and novel contexts.
</details>
<details>
<summary><u><strong>Exploring large language models for knowledge graph completion (2023)</strong></u></summary>

**Authors:** Liang Yao, Jiazhen Peng, Chengsheng Mao, Yuan Luo  
**Venue & Year:** IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2025), 2025  
**Link:** [https://doi.org/10.48550/arXiv.2308.13916](https://doi.org/10.48550/arXiv.2308.13916)  
**Abstract:**  
Knowledge graphs play a vital role in numerous artificial intelligence tasks, yet they frequently face the issue of incompleteness. In this study, we explore utilizing Large Language Models (LLM) for knowledge graph completion. We consider triples in knowledge graphs as text sequences and introduce an innovative framework called Knowledge Graph LLM (KG-LLM) to model these triples. Our technique employs entity and relation descriptions of a triple as prompts and utilizes the response for predictions. Experiments on various benchmark knowledge graphs demonstrate that our method attains state-of-the-art performance in tasks such as triple classification and relation prediction. We also find that fine-tuning relatively smaller models (e.g., LLaMA-7B, ChatGLM-6B) outperforms recent ChatGPT and GPT-4.
</details>
<details>
<summary><u><strong>KICGPT: Large Language Model with Knowledge in Context for Knowledge Graph Completion (2023)</strong></u></summary>

**Authors:** Yanbin Wei, Qiushi Huang, Yu Zhang, James Kwok  
**Venue & Year:** Findings of the Association for Computational Linguistics: EMNLP, 2023  
**Link:** [https://doi.org/10.18653/v1/2023.findings-emnlp.580](https://doi.org/10.18653/v1/2023.findings-emnlp.580)  
**Abstract:**  
Knowledge Graph Completion (KGC) is crucial for addressing knowledge graph incompleteness and supporting downstream applications. Many models have been proposed for KGC and they can be categorized into two main classes, including triple-based and test-based approaches. Triple-based methods struggle with long-tail entities due to limited structural information and imbalanced distributions of entities. Text-based methods alleviate this issue but require costly training for language models and specific finetuning for knowledge graphs, which limits their efficiency. To alleviate the limitations in the two approaches, in this paper, we propose KICGPT, a framework that integrates a large language model (LLM) and a triple-based KGC retriever, to alleviate the long-tail problem without incurring additional training overhead. In the proposed KICGPT model, we propose an in-context learning strategy called Knowledge Prompt, which encodes structural knowledge into demonstrations to guide LLM. Empirical results on benchmark datasets demonstrate the effectiveness of the proposed KICGPT model with lighter training overhead and no finetuning.

</details>
<details>
<summary><u><strong>Knowledge graph completion models are few-shot learners: An empirical study of relation labeling in e-commerce with LLMs (2023)</strong></u></summary>

**Authors:** Jiao Chen, Luyi Ma, Xiaohan Li, Nikhil Thakurdesai, Jianpeng Xu, Jason HD Cho, Kaushiki Nag, Evren Korpeoglu, Sushant Kumar, Kannan Achan  
**Venue & Year:** arXiv preprint, 2023  
**Link:** [https://doi.org/10.48550/arXiv.2305.09858](https://doi.org/10.48550/arXiv.2305.09858)  
**Abstract:**  
Knowledge Graphs (KGs) play a crucial role in enhancing e-commerce system performance by providing structured information about entities and their relationships, such as complementary or substitutable relations between products or product types, which can be utilized in recommender systems. However, relation labeling in KGs remains a challenging task due to the dynamic nature of e-commerce domains and the associated cost of human labor. Recently, breakthroughs in Large Language Models (LLMs) have shown surprising results in numerous natural language processing tasks. In this paper, we conduct an empirical study of LLMs for relation labeling in e-commerce KGs, investigating their powerful learning capabilities in natural language and effectiveness in predicting relations between product types with limited labeled data. We evaluate various LLMs, including PaLM and GPT-3.5, on benchmark datasets, demonstrating their ability to achieve competitive performance compared to humans on relation labeling tasks using just 1 to 5 labeled examples per relation. Additionally, we experiment with different prompt engineering techniques to examine their impact on model performance. Our results show that LLMs significantly outperform existing KG completion models in relation labeling for e-commerce KGs and exhibit performance strong enough to replace human labeling.
</details>
<details>
<summary><u><strong>Generate-on-graph: Treat LLM as both agent and KG in incomplete knowledge graph question answering (2024)</strong></u></summary>

**Authors:** Yao Xu, Shizhu He, Jiabei Chen, Zihao Wang, Yangqiu Song, Hanghang Tong, Guang Liu, Kang Liu, Jun Zhao  
**Venue & Year:** EMNLP, 2024  
**Link:** [https://doi.org/10.48550/arXiv.2404.14741](https://doi.org/10.48550/arXiv.2404.14741)  
**Abstract:**  
To address the issues of insufficient knowledge and hallucination in Large Language Models (LLMs), numerous studies have explored integrating LLMs with Knowledge Graphs (KGs). However, these methods are typically evaluated on conventional Knowledge Graph Question Answering (KGQA) with complete KGs, where all factual triples required for each question are entirely covered by the given KG. In such cases, LLMs primarily act as an agent to find answer entities within the KG, rather than effectively integrating the internal knowledge of LLMs and external knowledge sources such as KGs. In fact, KGs are often incomplete to cover all the knowledge required to answer questions. To simulate these real-world scenarios and evaluate the ability of LLMs to integrate internal and external knowledge, we propose leveraging LLMs for QA under Incomplete Knowledge Graph (IKGQA), where the provided KG lacks some of the factual triples for each question, and construct corresponding datasets. To handle IKGQA, we propose a training-free method called Generate-on-Graph (GoG), which can generate new factual triples while exploring KGs. Specifically, GoG performs reasoning through a Thinking-Searching-Generating framework, which treats LLM as both Agent and KG in IKGQA. Experimental results on two datasets demonstrate that our GoG outperforms all previous methods.
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
**Abstract:**  
As the market for illicit drugs remains extremely profitable, major online platforms have become direct-to-consumer intermediaries for illicit drug trafficking participants. These online activities raise significant social concerns that require immediate actions. Existing approaches to combating this challenge are generally impractical, due to the imbalance of classes and scarcity of labeled samples in real-world applications. To this end, we propose a novel Large Language Model-empowered Heterogeneous Graph Prompt Learning framework for illicit Drug Trafficking detection, called LLM-HetGDT, that leverages LLM to facilitate heterogeneous graph neural networks (HGNNs) to effectively identify drug trafficking activities in the class-imbalanced scenarios. Specifically, we first pre-train HGNN over a contrastive pretext task to capture the inherent node and structure information over the unlabeled drug trafficking heterogeneous graph (HG). Afterward, we employ LLM to augment the HG by generating high-quality synthetic user nodes in minority classes. Then, we fine-tune the soft prompts on the augmented HG to capture the important information in the minority classes for the downstream drug trafficking detection task. To comprehensively study online illicit drug trafficking activities, we collect a new HG dataset over Twitter, called Twitter-HetDrug. Extensive experiments on this dataset demonstrate the effectiveness, efficiency, and applicability of LLM-HetGDT.
</details>
<details>
<summary><u><strong>Large language model assisted fine-grained knowledge graph construction for robotic fault diagnosis (2025)</strong></u></summary>

**Authors:** Xingming Liao, Chong Chen, Zhuowei Wang, Ying Liu, Tao Wang, Lianglun Cheng  
**Venue & Year:** Advanced Engineering Informatics, 2025  
**Link:** [https://doi.org/10.1016/j.aei.2025.103134](https://doi.org/10.1016/j.aei.2025.103134)  
**Abstract:**  
With the rapid deployment of industrial robots in manufacturing, the demand for advanced maintenance techniques to sustain operational efficiency has become crucial. Fault diagnosis Knowledge Graph (KG) is essential as it interlinks multi-source data related to industrial robot faults, capturing multi-level semantic associations among different fault events. However, the construction and application of fine-grained fault diagnosis KG face significant challenges due to the inherent complexity of nested entities in maintenance texts and the severe scarcity of annotated industrial data. In this study, we propose a Large Language Model (LLM) assisted data augmentation approach, which handles the complex nested entities in maintenance corpora and constructs a more fine-grained fault diagnosis KG. Firstly, the fine-grained ontology is constructed via LLM Assistance in Industrial Nested Named Entity Recognition (assInNNER). Then, an Industrial Nested Label Classification Template (INCT) is designed, enabling the use of nested entities in Attention-map aware keyword selection for the Industrial Nested Language Model (ANLM) data augmentation methods. ANLM can effectively improve the model’s performance in nested entity extraction when corpora are scarce. Subsequently, a Confidence Filtering Mechanism (CFM) is introduced to evaluate and select the generated data for enhancement, and assInNNER is further deployed to recall the negative samples corpus again to further improve performance. Experimental studies based on multi-source corpora demonstrate that compared to existing algorithms, our method achieves an average F1 increase of 8.25 %, 3.31 %, and 1.96 % in 5%, 10 %, and 25 % in few-shot settings, respectively.
</details>
<details>
<summary><u><strong>Low-resource knowledge graph completion based on knowledge distillation driven by large language models (2025)</strong></u></summary>

**Authors:** Wenlong Hou, Weidong Zhao, Ning Jia, Xianhui Liu  
**Venue & Year:** Applied Soft Computing, 2025  
**Link:** [https://doi.org/10.1016/j.asoc.2024.112622](https://doi.org/10.1016/j.asoc.2024.112622)  
**Abstract:**  
Knowledge graph completion (KGC) refines the existing knowledge graph (KG) by predicting missing entities or relations. Existing methods are mainly based on embeddings or texts but only perform better with abundant labeled data. Hence, KGC in resource-constrained settings is a significant problem, which faces challenges of data imbalance across relations and lack of relation label semantics. Considering that Large Language Models (LLMs) demonstrate powerful reasoning and generation capabilities, this work proposes an LLM-driven Knowledge Graph Completion Distillation (KGCD) model to address low-resource KGC. A two-stage framework is developed, involving teacher-student distillation by using LLM to improve reasoning, followed by fine-tuning on real-world low-resource datasets. To deal with data imbalance, a hybrid prompt design for LLM is proposed, which includes rethink and open prompts. Furthermore, a virtual relation label generation strategy enhances the model’s understanding of triples. Extensive experiments on three benchmarks have shown that KGCD’s effectiveness for low-resource KGC, achieving improvements in Mean Reciprocal Rank (MRR) by 11% and Hits@1 by 10% on the WN18, MRR by 10% and Hits@1 by 14% on the WN18RR, and MRR by 12% and Hits@1 by 11% on the YAGO3-10.
</details>
<details>
<summary><u><strong>Empowering graph neural network-based computational drug repositioning with large language model-inferred knowledge representation (2024)</strong></u></summary>

**Authors:** Yaowen Gu, Zidu Xu, Carl Yang  
**Venue & Year:** Interdisciplinary Sciences: Computational Life Sciences, 2024  
**Link:** [https://doi.org/10.1007/s12539-024-00654-7](https://doi.org/10.1007/s12539-024-00654-7)  
**Abstract:**  
Computational drug repositioning, through predicting drug-disease associations (DDA), offers significant potential for discovering new drug indications. Current methods incorporate graph neural networks (GNN) on drug-disease heterogeneous networks to predict DDAs, achieving notable performances compared to traditional machine learning and matrix factorization approaches. However, these methods depend heavily on network topology, hampered by incomplete and noisy network data, and overlook the wealth of biomedical knowledge available. Correspondingly, large language models (LLMs) excel in graph search and relational reasoning, which can possibly enhance the integration of comprehensive biomedical knowledge into drug and disease profiles. In this study, we first investigate the contribution of LLM-inferred knowledge representation in drug repositioning and DDA prediction. A zero-shot prompting template was designed for LLM to extract high-quality knowledge descriptions for drug and disease entities, followed by embedding generation from language models to transform the discrete text to continual numerical representation. Then, we proposed LLM-DDA with three different model architectures (LLM-DDANode Feat, LLM-DDADual GNN, LLM-DDAGNN-AE) to investigate the best fusion mode for LLM-based embeddings. Extensive experiments on four DDA benchmarks show that, LLM-DDAGNN-AE achieved the optimal performance compared to 11 baselines with the overall relative improvement in AUPR of 23.22%, F1-Score of 17.20%, and precision of 25.35%. Meanwhile, selected case studies of involving Prednisone and Allergic Rhinitis highlighted the model’s capability to identify reliable DDAs and knowledge descriptions, supported by existing literature. This study showcases the utility of LLMs in drug repositioning with its generality and applicability in other biomedical relation prediction tasks.
</details>
<details>
<summary><u><strong>Cost-Effective Label-free Node Classification with LLMs (2024)</strong></u></summary>

**Authors:** Taiyan Zhang, Renchi Yang, Mingyu Yan, Xiaochun Ye, Dongrui Fan, Yurui Lai  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2412.11983](https://arxiv.org/abs/2412.11983)  
**Abstract:**  
Graph neural networks (GNNs) have become the preferred models for node classification in graph data due to their robust capabilities in integrating graph structures and attributes. However, these models heavily depend on a substantial amount of high-quality labeled data for training, which is often costly to obtain. With the rise of large language models (LLMs), a promising approach is to utilize their exceptional zero-shot capabilities and extensive knowledge for node labeling. Despite encouraging results, this approach either requires numerous queries to LLMs or suffers from reduced performance due to noisy labels generated by LLMs. To address these challenges, we introduce Locle, an active self-training framework that does Label-free node Classification with LLMs cost-Effectively. Locle iteratively identifies small sets of "critical" samples using GNNs and extracts informative pseudo-labels for them with both LLMs and GNNs, serving as additional supervision signals to enhance model training. Specifically, Locle comprises three key components: (i) an effective active node selection strategy for initial annotations; (ii) a careful sample selection scheme to identify "critical" nodes based on label disharmonicity and entropy; and (iii) a label refinement module that combines LLMs and GNNs with a rewired topology. Extensive experiments on five benchmark text-attributed graph datasets demonstrate that Locle significantly outperforms state-of-the-art methods under the same query budget to LLMs in terms of label-free node classification. Notably, on the DBLP dataset with 14.3k nodes, Locle achieves an 8.08% improvement in accuracy over the state-of-the-art at a cost of less than one cent. Our code is available at [https://github.com/HKBU-LAGAS/Locle](https://github.com/HKBU-LAGAS/Locle)
</details>
<details>
<summary><u><strong>Enhancing student performance prediction on learnersourced questions with SGNN-LLM synergy (2024)</strong></u></summary>

**Authors:** Lin Ni, Sijie Wang, Zeyu Zhang, Xiaoxuan Li, Xianda Zheng, Paul Denny, Jiamou Liu  
**Venue & Year:** Proceedings of the AAAI Conference on Artificial Intelligence, 2024  
**Link:** [https://doi.org/10.1609/aaai.v38i21.30370](https://doi.org/10.1609/aaai.v38i21.30370)  
**Abstract:**  
Learnersourcing offers great potential for scalable education through student content creation. However, predicting student performance on learnersourced questions, which is essential for personalizing the learning experience, is challenging due to the inherent noise in student-generated data. Moreover, while conventional graph-based methods can capture the complex network of student and question interactions, they often fall short under cold start conditions where limited student engagement with questions yields sparse data. To address both challenges, we introduce an innovative strategy that synergizes the potential of integrating Signed Graph Neural Networks (SGNNs) and Large Language Model (LLM) embeddings. Our methodology employs a signed bipartite graph to comprehensively model student answers, complemented by a contrastive learning framework that enhances noise resilience. Furthermore, LLM's contribution lies in generating foundational question embeddings, proving especially advantageous in addressing cold start scenarios characterized by limited graph data. Validation across five real-world datasets sourced from the PeerWise platform underscores our approach's effectiveness. Our method outperforms baselines, showcasing enhanced predictive accuracy and robustness.
</details>
<details>
<summary><u><strong>Depression detection in clinical interviews with LLM-empowered structural element graph (2024)</strong></u></summary>

**Authors:** Zhuang Chen, Jiawen Deng, Jinfeng Zhou, Jincenzi Wu, Tieyun Qian, Minlie Huang  
**Venue & Year:** Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2024  
**Link:** [https://aclanthology.org/2024.naacl-long.452/](https://aclanthology.org/2024.naacl-long.452/)  
**Abstract:**  
Depression is a widespread mental health disorder affecting millions globally. Clinical interviews are the gold standard for assessing depression, but they heavily rely on scarce professional clinicians, highlighting the need for automated detection systems. However, existing methods only capture part of the relevant elements in clinical interviews, unable to incorporate all depressive cues. Moreover, the scarcity of participant data, due to privacy concerns and collection challenges, intrinsically constrains interview modeling. To address these limitations, in this paper, we propose a structural element graph (SEGA), which transforms the clinical interview into an expertise-inspired directed acyclic graph for comprehensive modeling. Additionally, we further empower SEGA by devising novel principle-guided data augmentation with large language models (LLMs) to supplement high-quality synthetic data and enable graph contrastive learning. Extensive evaluations on two real-world clinical datasets, in both English and Chinese, show that SEGA significantly outperforms baseline methods and powerful LLMs like GPT-3.5 and GPT-4.
</details>
<details>
<summary><u><strong>Fine-grainedly Synthesize Streaming Data Based On Large Language Models With Graph Structure Understanding For Data Sparsity (2024)</strong></u></summary>

**Authors:** Xin Zhang, Linhai Zhang, Deyu Zhou, Guoqiang Xu  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2403.06139](https://arxiv.org/abs/2403.06139)  
**Abstract:**  
Due to the sparsity of user data, sentiment analysis on user reviews in e-commerce platforms often suffers from poor performance, especially when faced with extremely sparse user data or long-tail labels. Recently, the emergence of LLMs has introduced new solutions to such problems by leveraging graph structures to generate supplementary user profiles. However, previous approaches have not fully utilized the graph understanding capabilities of LLMs and have struggled to adapt to complex streaming data environments. In this work, we propose a fine-grained streaming data synthesis framework that categorizes sparse users into three categories: Mid-tail, Long-tail, and Extreme. Specifically, we design LLMs to comprehensively understand three key graph elements in streaming data, including Local-global Graph Understanding, Second-Order Relationship Extraction, and Product Attribute Understanding, which enables the generation of high-quality synthetic data to effectively address sparsity across different categories. Experimental results on three real datasets demonstrate significant performance improvements, with synthesized data contributing to MSE reductions of 45.85%, 3.16%, and 62.21%, respectively.
</details>
<details>
<summary><u><strong>Large Language Model-based Augmentation for Imbalanced Node Classification on Text-Attributed Graphs (2024)</strong></u></summary>

**Authors:** Leyao Wang, Yu Wang, Bo Ni, Yuying Zhao, Tyler Derr  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2410.16882](https://arxiv.org/abs/2410.16882)  
**Abstract:**  
Node classification on graphs frequently encounters the challenge of class imbalance, leading to biased performance and posing significant risks in real-world applications. Although several data-centric solutions have been proposed, none of them focus on Text-Attributed Graphs (TAGs), and therefore overlook the potential of leveraging the rich semantics encoded in textual features for boosting the classification of minority nodes. Given this crucial gap, we investigate the possibility of augmenting graph data in the text space, leveraging the textual generation power of Large Language Models (LLMs) to handle imbalanced node classification on TAGs. Specifically, we propose a novel approach called LA-TAG (LLM-based Augmentation on Text-Attributed Graphs), which prompts LLMs to generate synthetic texts based on existing node texts in the graph. Furthermore, to integrate these synthetic text-attributed nodes into the graph, we introduce a text-based link predictor to connect the synthesized nodes with the existing nodes. Our experiments across multiple datasets and evaluation metrics show that our framework significantly outperforms traditional non-textual-based data augmentation strategies and specific node imbalance solutions. This highlights the promise of using LLMs to resolve imbalance issues on TAGs.
</details>
<details>
<summary><u><strong>Distilling large language models for text-attributed graph learning (2024)</strong></u></summary>

**Authors:** Bo Pan, Zheng Zhang, Yifei Zhang, Yuntong Hu, Liang Zhao  
**Venue & Year:** Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, 2024  
**Link:** [https://doi.org/10.1145/3627673.3679830](https://doi.org/10.1145/3627673.3679830)  
**Abstract:**  
Text-Attributed Graphs (TAGs) are graphs of connected textual documents. Graph models can efficiently learn TAGs, but their training heavily relies on human-annotated labels, which are scarce or even unavailable in many applications. Large language models (LLMs) have recently demonstrated remarkable capabilities in few-shot and zero-shot TAG learning, but they suffer from scalability, cost, and privacy issues. Therefore, in this work, we focus on synergizing LLMs and graph models with their complementary strengths by distilling the power of LLMs into a local graph model on TAG learning. To address the inherent gaps between LLMs (generative models for texts) and graph models (discriminative models for graphs), we propose first to let LLMs teach an interpreter with rich rationale and then let a student model mimic the interpreter's reasoning without LLMs' rationale. We convert LLM's textual rationales to multi-level graph rationales to train the interpreter model and align the student model with the interpreter model based on the features of TAGs. Extensive experiments validate the efficacy of our proposed framework.
</details>
<details>
<summary><u><strong>KICGPT: Large Language Model with Knowledge in Context for Knowledge Graph Completion (2023)</strong></u></summary>

**Authors:** Yanbin Wei, Qiushi Huang, Yu Zhang, James Kwok  
**Venue & Year:** Findings of the Association for Computational Linguistics: EMNLP, 2023  
**Link:** [https://doi.org/10.18653/v1/2023.findings-emnlp.580](https://doi.org/10.18653/v1/2023.findings-emnlp.580)  
**Abstract:**  
Knowledge Graph Completion (KGC) is crucial for addressing knowledge graph incompleteness and supporting downstream applications. Many models have been proposed for KGC and they can be categorized into two main classes, including triple-based and test-based approaches. Triple-based methods struggle with long-tail entities due to limited structural information and imbalanced distributions of entities. Text-based methods alleviate this issue but require costly training for language models and specific finetuning for knowledge graphs, which limits their efficiency. To alleviate the limitations in the two approaches, in this paper, we propose KICGPT, a framework that integrates a large language model (LLM) and a triple-based KGC retriever, to alleviate the long-tail problem without incurring additional training overhead. In the proposed KICGPT model, we propose an in-context learning strategy called Knowledge Prompt, which encodes structural knowledge into demonstrations to guide LLM. Empirical results on benchmark datasets demonstrate the effectiveness of the proposed KICGPT model with lighter training overhead and no finetuning.
</details>
<details>
<summary><u><strong>Label-free node classification on graphs with large language models (LLMs) (2023)</strong></u></summary>

**Authors:** Zhikai Chen, Haitao Mao, Hongzhi Wen, Haoyu Han, Wei Jin, Haiyang Zhang, Hui Liu, Jiliang Tang  
**Venue & Year:** arXiv preprint, 2023  
**Link:** [https://arxiv.org/abs/2310.04668](https://arxiv.org/abs/2310.04668)  
**Abstract:**  
In recent years, there have been remarkable advancements in node classification achieved by Graph Neural Networks (GNNs). However, they necessitate abundant high-quality labels to ensure promising performance. In contrast, Large Language Models (LLMs) exhibit impressive zero-shot proficiency on text-attributed graphs. Yet, they face challenges in efficiently processing structural data and suffer from high inference costs. In light of these observations, this work introduces a label-free node classification on graphs with LLMs pipeline, LLM-GNN. It amalgamates the strengths of both GNNs and LLMs while mitigating their limitations. Specifically, LLMs are leveraged to annotate a small portion of nodes and then GNNs are trained on LLMs' annotations to make predictions for the remaining large portion of nodes. The implementation of LLM-GNN faces a unique challenge: how can we actively select nodes for LLMs to annotate and consequently enhance the GNN training? How can we leverage LLMs to obtain annotations of high quality, representativeness, and diversity, thereby enhancing GNN performance with less cost? To tackle this challenge, we develop an annotation quality heuristic and leverage the confidence scores derived from LLMs to advanced node selection. Comprehensive experimental results validate the effectiveness of LLM-GNN. In particular, LLM-GNN can achieve an accuracy of 74.9% on a vast-scale dataset \products with a cost less than 1 dollar.
</details>
<details>
<summary><u><strong>Augmenting low-resource text classification with graph-grounded pre-training and prompting (2023)</strong></u></summary>

**Authors:** Zhihao Wen, Yuan Fang  
**Venue & Year:** Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2023  
**Link:** [https://doi.org/10.1145/3539618.3591641](https://doi.org/10.1145/3539618.3591641)  
**Abstract:**  
Text classification is a fundamental problem in information retrieval with many real-world applications, such as predicting the topics of online articles and the categories of e-commerce product descriptions. However, low-resource text classification, with few or no labeled samples, poses a serious concern for supervised learning. Meanwhile, many text data are inherently grounded on a network structure, such as a hyperlink/citation network for online articles, and a user-item purchase network for e-commerce products. These graph structures capture rich semantic relationships, which can potentially augment low-resource text classification. In this paper, we propose a novel model called Graph-Grounded Pre-training and Prompting (G2P2) to address low-resource text classification in a two-pronged approach. During pre-training, we propose three graph interaction-based contrastive strategies to jointly pre-train a graph-text model; during downstream classification, we explore prompting for the jointly pre-trained model to achieve low-resource classification. Extensive experiments on four real-world datasets demonstrate the strength of G2P2 in zero- and few-shot low-resource text classification tasks.
</details>
<details>
<summary><u><strong>Leveraging Large Language Models for Node Generation in Few-Shot Learning on Text-Attributed Graphs (2023)</strong></u></summary>

**Authors:** Jianxiang Yu, Yuxiang Ren, Chenghua Gong, Jiaqi Tan, Xiang Li, Xuecang Zhang  
**Venue & Year:** arXiv preprints, 2023  
**Link:** [https://arxiv.org/abs/2310.09872](https://arxiv.org/abs/2310.09872)  
**Abstract:**  
Text-attributed graphs have recently garnered significant attention due to their wide range of applications in web domains. Existing methodologies employ word embedding models for acquiring text representations as node features, which are subsequently fed into Graph Neural Networks (GNNs) for training. Recently, the advent of Large Language Models (LLMs) has introduced their powerful capabilities in information retrieval and text generation, which can greatly enhance the text attributes of graph data. Furthermore, the acquisition and labeling of extensive datasets are both costly and time-consuming endeavors. Consequently, few-shot learning has emerged as a crucial problem in the context of graph learning tasks. In order to tackle this challenge, we propose a lightweight paradigm called LLM4NG, which adopts a plug-and-play approach to empower text-attributed graphs through node generation using LLMs. Specifically, we utilize LLMs to extract semantic information from the labels and generate samples that belong to these categories as exemplars. Subsequently, we employ an edge predictor to capture the structural information inherent in the raw dataset and integrate the newly generated samples into the original graph. This approach harnesses LLMs for enhancing class-level information and seamlessly introduces labeled nodes and edges without modifying the raw dataset, thereby facilitating the node classification task in few-shot scenarios. Extensive experiments demonstrate the outstanding performance of our proposed paradigm, particularly in low-shot scenarios. For instance, in the 1-shot setting of the ogbn-arxiv dataset, LLM4NG achieves a 76% improvement over the baseline model.
</details>
<details>
<summary><u><strong>LKPNR: LLM and KG for Personalized News Recommendation Framework (2023)</strong></u></summary>

**Authors:** Xie Runfeng, Cui Xiangyang, Yan Zhou, Wang Xin, Xuan Zhanwei, Zhang Kai, et al.  
**Venue & Year:** arXiv preprint, 2023  
**Link:** [https://arxiv.org/abs/2308.12028](https://arxiv.org/abs/2308.12028)  
**Abstract:**  
Accurately recommending candidate news articles to users is a basic challenge faced by personalized news recommendation systems. Traditional methods are usually difficult to grasp the complex semantic information in news texts, resulting in unsatisfactory recommendation results. Besides, these traditional methods are more friendly to active users with rich historical behaviors. However, they can not effectively solve the "long tail problem" of inactive users. To address these issues, this research presents a novel general framework that combines Large Language Models (LLM) and Knowledge Graphs (KG) into semantic representations of traditional methods. In order to improve semantic understanding in complex news texts, we use LLMs' powerful text understanding ability to generate news representations containing rich semantic information. In addition, our method combines the information about news entities and mines high-order structural information through multiple hops in KG, thus alleviating the challenge of long tail distribution. Experimental results demonstrate that compared with various traditional models, the framework significantly improves the recommendation effect. The successful integration of LLM and KG in our framework has established a feasible path for achieving more accurate personalized recommendations in the news field. Our code is available at this https URL[https://github.com/Xuan-ZW/LKPNR](https://github.com/Xuan-ZW/LKPNR).
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
**Abstract:**  
Graph neural networks (GNNs) are vulnerable to adversarial attacks, especially for topology perturbations, and many methods that improve the robustness of GNNs have received considerable attention. Recently, we have witnessed the significant success of large language models (LLMs), leading many to explore the great potential of LLMs on GNNs. However, they mainly focus on improving the performance of GNNs by utilizing LLMs to enhance the node features. Therefore, we ask: Will the robustness of GNNs also be enhanced with the powerful understanding and inference capabilities of LLMs? By presenting the empirical results, we find that despite that LLMs can improve the robustness of GNNs, there is still an average decrease of 23.1% in accuracy, implying that the GNNs remain extremely vulnerable against topology attacks. Therefore, another question is how to extend the capabilities of LLMs on graph adversarial robustness. In this paper, we propose an LLM-based robust graph structure inference framework, LLM4RGNN, which distills the inference capabilities of GPT-4 into a local LLM for identifying malicious edges and an LM-based edge predictor for finding missing important edges, so as to recover a robust graph structure. Extensive experiments demonstrate that LLM4RGNN consistently improves the robustness across various GNNs. Even in some cases where the perturbation ratio increases to 40%, the accuracy of GNNs is still better than that on the clean graph. The source code can be found in [https://github.com/zhongjian-zhang/LLM4RGNN](https://github.com/zhongjian-zhang/LLM4RGNN).
</details>
<details>
<summary><u><strong>Subgraph-Aware Training of Language Models for Knowledge Graph Completion Using Structure-Aware Contrastive Learning (2024)</strong></u></summary>

**Authors:** Youmin Ko, Hyemin Yang, Taeuk Kim, Hyunjoon Kim  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2407.12703](https://arxiv.org/abs/2407.12703)  
**Abstract:**  
Fine-tuning pre-trained language models (PLMs) has recently shown a potential to improve knowledge graph completion (KGC). However, most PLM-based methods focus solely on encoding textual information, neglecting the long-tailed nature of knowledge graphs and their various topological structures, e.g., subgraphs, shortest paths, and degrees. We claim that this is a major obstacle to achieving higher accuracy of PLMs for KGC. To this end, we propose a Subgraph-Aware Training framework for KGC (SATKGC) with two ideas: (i) subgraph-aware mini-batching to encourage hard negative sampling and to mitigate an imbalance in the frequency of entity occurrences during training, and (ii) new contrastive learning to focus more on harder in-batch negative triples and harder positive triples in terms of the structural properties of the knowledge graph. To the best of our knowledge, this is the first study to comprehensively incorporate the structural inductive bias of the knowledge graph into fine-tuning PLMs. Extensive experiments on three KGC benchmarks demonstrate the superiority of SATKGC. Our code is available.
</details>
<details>
<summary><u><strong>Multi-perspective improvement of knowledge graph completion with large language models (2024)</strong></u></summary>

**Authors:** Derong Xu, Ziheng Zhang, Zhenxi Lin, Xian Wu, Zhihong Zhu, Tong Xu, Xiangyu Zhao, Yefeng Zheng, Enhong Chen  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2403.01972](https://arxiv.org/abs/2403.01972)  
**Abstract:**  
Knowledge graph completion (KGC) is a widely used method to tackle incompleteness in knowledge graphs (KGs) by making predictions for missing links. Description-based KGC leverages pre-trained language models to learn entity and relation representations with their names or descriptions, which shows promising results. However, the performance of description-based KGC is still limited by the quality of text and the incomplete structure, as it lacks sufficient entity descriptions and relies solely on relation names, leading to sub-optimal results. To address this issue, we propose MPIKGC, a general framework to compensate for the deficiency of contextualized knowledge and improve KGC by querying large language models (LLMs) from various perspectives, which involves leveraging the reasoning, explanation, and summarization capabilities of LLMs to expand entity descriptions, understand relations, and extract structures, respectively. We conducted extensive evaluation of the effectiveness and improvement of our framework based on four description-based KGC models and four datasets, for both link prediction and triplet classification tasks.
</details>
<details>
<summary><u><strong>Graphedit: Large language models for graph structure learning (2024)</strong></u></summary>

**Authors:** Zirui Guo, Lianghao Xia, Yanhua Yu, Yuling Wang, Zixuan Yang, Wei Wei, Liang Pang, Tat-Seng Chua, Chao Huang  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2402.15183](https://arxiv.org/abs/2402.15183)  
**Abstract:**  
Graph Structure Learning (GSL) focuses on capturing intrinsic dependencies and interactions among nodes in graph-structured data by generating novel graph structures. Graph Neural Networks (GNNs) have emerged as promising GSL solutions, utilizing recursive message passing to encode node-wise inter-dependencies. However, many existing GSL methods heavily depend on explicit graph structural information as supervision signals, leaving them susceptible to challenges such as data noise and sparsity. In this work, we propose GraphEdit, an approach that leverages large language models (LLMs) to learn complex node relationships in graph-structured data. By enhancing the reasoning capabilities of LLMs through instruction-tuning over graph structures, we aim to overcome the limitations associated with explicit graph structural information and enhance the reliability of graph structure learning. Our approach not only effectively denoises noisy connections but also identifies node-wise dependencies from a global perspective, providing a comprehensive understanding of the graph structure. We conduct extensive experiments on multiple benchmark datasets to demonstrate the effectiveness and robustness of GraphEdit across various settings. We have made our model implementation available at: [https://github.com/HKUDS/GraphEdit](https://github.com/HKUDS/GraphEdit).
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
**Abstract:**  
Hypergraph neural networks (HGNNs) effectively model complex high-order relationships in domains like protein interactions and social networks by connecting multiple vertices through hyperedges, enhancing modeling capabilities, and reducing information loss. Developing foundation models for hypergraphs is challenging due to their distinct data, which includes both vertex features and intricate structural information. We present Hyper-FM, a Hypergraph Foundation Model for multi-domain knowledge extraction, featuring Hierarchical High-Order Neighbor Guided Vertex Knowledge Embedding for vertex feature representation and Hierarchical Multi-Hypergraph Guided Structural Knowledge Extraction for structural information. Additionally, we curate 10 text-attributed hypergraph datasets to advance research between HGNNs and LLMs. Experiments on these datasets show that Hyper-FM outperforms baseline methods by approximately 13.3%, validating our approach. Furthermore, we propose the first scaling law for hypergraph foundation models, demonstrating that increasing domain diversity significantly enhances performance, unlike merely augmenting vertex and hyperedge counts. This underscores the critical role of domain diversity in scaling hypergraph models.
</details>
<details>
<summary><u><strong>UniGraph: Learning a Cross-Domain Graph Foundation Model From Natural Language (2024)</strong></u></summary>

**Authors:** Yufei He, Bryan Hooi  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2402.13630](https://arxiv.org/abs/2402.13630)  
**Abstract:**  
Foundation models like ChatGPT and GPT-4 have revolutionized artificial intelligence, exhibiting remarkable abilities to generalize across a wide array of tasks and applications beyond their initial training objectives. However, graph learning has predominantly focused on single-graph models, tailored to specific tasks or datasets, lacking the ability to transfer learned knowledge to different domains. This limitation stems from the inherent complexity and diversity of graph structures, along with the different feature and label spaces specific to graph data. In this paper, we recognize text as an effective unifying medium and employ Text-Attributed Graphs (TAGs) to leverage this potential. We present our UniGraph framework, designed to learn a foundation model for TAGs, which is capable of generalizing to unseen graphs and tasks across diverse domains. Unlike single-graph models that use pre-computed node features of varying dimensions as input, our approach leverages textual features for unifying node representations, even for graphs such as molecular graphs that do not naturally have textual features. We propose a novel cascaded architecture of Language Models (LMs) and Graph Neural Networks (GNNs) as backbone networks. Additionally, we propose the first pre-training algorithm specifically designed for large-scale self-supervised learning on TAGs, based on Masked Graph Modeling. We introduce graph instruction tuning using Large Language Models (LLMs) to enable zero-shot prediction ability. Our comprehensive experiments across various graph learning tasks and domains demonstrate the model's effectiveness in self-supervised representation learning on unseen graphs, few-shot in-context transfer, and zero-shot transfer, even surpassing or matching the performance of GNNs that have undergone supervised training on target datasets.
</details>
<details>
<summary><u><strong>LLM-Align: Utilizing Large Language Models for Entity Alignment in Knowledge Graphs (2024)</strong></u></summary>

**Authors:** Xuan Chen, Tong Lu, Zhichun Wang  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2412.04690](https://arxiv.org/abs/2412.04690)  
**Abstract:**  
Entity Alignment (EA) seeks to identify and match corresponding entities across different Knowledge Graphs (KGs), playing a crucial role in knowledge fusion and integration. Embedding-based entity alignment (EA) has recently gained considerable attention, resulting in the emergence of many innovative approaches. Initially, these approaches concentrated on learning entity embeddings based on the structural features of knowledge graphs (KGs) as defined by relation triples. Subsequent methods have integrated entities' names and attributes as supplementary information to improve the embeddings used for EA. However, existing methods lack a deep semantic understanding of entity attributes and relations. In this paper, we propose a Large Language Model (LLM) based Entity Alignment method, LLM-Align, which explores the instruction-following and zero-shot capabilities of Large Language Models to infer alignments of entities. LLM-Align uses heuristic methods to select important attributes and relations of entities, and then feeds the selected triples of entities to an LLM to infer the alignment results. To guarantee the quality of alignment results, we design a multi-round voting mechanism to mitigate the hallucination and positional bias issues that occur with LLMs. Experiments on three EA datasets, demonstrating that our approach achieves state-of-the-art performance compared to existing EA methods.
</details>
<details>
<summary><u><strong>Bootstrapping Heterogeneous Graph Representation Learning via Large Language Models: A Generalized Approach (2024)</strong></u></summary>

**Authors:** Hang Gao, Chenhao Zhang, Fengge Wu, Junsuo Zhao, Changwen Zheng, Huaping Liu  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2412.08038](https://arxiv.org/abs/2412.08038)  
**Abstract:**  
Graph representation learning methods are highly effective in handling complex non-Euclidean data by capturing intricate relationships and features within graph structures. However, traditional methods face challenges when dealing with heterogeneous graphs that contain various types of nodes and edges due to the diverse sources and complex nature of the data. Existing Heterogeneous Graph Neural Networks (HGNNs) have shown promising results but require prior knowledge of node and edge types and unified node feature formats, which limits their applicability. Recent advancements in graph representation learning using Large Language Models (LLMs) offer new solutions by integrating LLMs' data processing capabilities, enabling the alignment of various graph representations. Nevertheless, these methods often overlook heterogeneous graph data and require extensive preprocessing. To address these limitations, we propose a novel method that leverages the strengths of both LLM and GNN, allowing for the processing of graph data with any format and type of nodes and edges without the need for type information or special preprocessing. Our method employs LLM to automatically summarize and classify different data formats and types, aligns node features, and uses a specialized GNN for targeted learning, thus obtaining effective graph representations for downstream tasks. Theoretical analysis and experimental validation have demonstrated the effectiveness of our method.
</details>
<details>
<summary><u><strong>Harnessing Explanations: LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning (2024)</strong></u></summary>

**Authors:** Xiaoxin He, Xavier Bresson, Thomas Laurent, Adam Perold, Yann LeCun, Bryan Hooi  
**Venue & Year:** International Conference on Learning Representations, 2024  
**Link:** [https://openreview.net/forum?id=RXFVcynVe1](https://openreview.net/forum?id=RXFVcynVe1)  
**Abstract:**  
Representation learning on text-attributed graphs (TAGs) has become a critical research problem in recent years. A typical example of a TAG is a paper citation graph, where the text of each paper serves as node attributes. Initial graph neural network (GNN) pipelines handled these text attributes by transforming them into shallow or hand-crafted features, such as skip-gram or bag-of-words features. Recent efforts have focused on enhancing these pipelines with language models (LMs), which typically demand intricate designs and substantial computational resources. With the advent of powerful large language models (LLMs) such as GPT or Llama2, which demonstrate an ability to reason and to utilize general knowledge, there is a growing need for techniques which combine the textual modelling abilities of LLMs with the structural learning capabilities of GNNs. Hence, in this work, we focus on leveraging LLMs to capture textual information as features, which can be used to boost GNN performance on downstream tasks. A key innovation is our use of explanations as features: we prompt an LLM to perform zero-shot classification, request textual explanations for its decision-making process, and design an LLM-to-LM interpreter to translate these explanations into informative features for downstream GNNs. Our experiments demonstrate that our method achieves state-of-the-art results on well-established TAG datasets, including Cora, PubMed, ogbn-arxiv, as well as our newly introduced dataset, tape-arxiv23. Furthermore, our method significantly speeds up training, achieving a 2.88 times improvement over the closest baseline on ogbn-arxiv. Lastly, we believe the versatility of the proposed method extends beyond TAGs and holds the potential to enhance other tasks involving graph-text data.
</details>
<details>
<summary><u><strong>LLMRec: Large Language Models with Graph Augmentation for Recommendation (2024)</strong></u></summary>

**Authors:** Wei Wei, Xubin Ren, Jiabin Tang, Qinyong Wang, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, Chao Huang  
**Venue & Year:** Proceedings of the 17th ACM International Conference on Web Search and Data Mining, 2024  
**Link:** [https://doi.org/10.1145/3616855.3635853](https://doi.org/10.1145/3616855.3635853)  
**Abstract:**  
The problem of data sparsity has long been a challenge in recommendation systems, and previous studies have attempted to address this issue by incorporating side information. However, this approach often introduces side effects such as noise, availability issues, and low data quality, which in turn hinder the accurate modeling of user preferences and adversely impact recommendation performance. In light of the recent advancements in large language models (LLMs), which possess extensive knowledge bases and strong reasoning capabilities, we propose a novel framework called LLMRec that enhances recommender systems by employing three simple yet effective LLM-based graph augmentation strategies. Our approach leverages the rich content available within online platforms (e.g., Netflix, MovieLens) to augment the interaction graph in three ways: (i) reinforcing user-item interaction egde, (ii) enhancing the understanding of item node attributes, and (iii) conducting user node profiling, intuitively from the natural language perspective. By employing these strategies, we address the challenges posed by sparse implicit feedback and low-quality side information in recommenders. Besides, to ensure the quality of the augmentation, we develop a denoised data robustification mechanism that includes techniques of noisy implicit feedback pruning and MAE-based feature enhancement that help refine the augmented data and improve its reliability. Furthermore, we provide theoretical analysis to support the effectiveness of LLMRec and clarify the benefits of our method in facilitating model optimization. Experimental results on benchmark datasets demonstrate the superiority of our LLM-based augmentation approach over state-of-the-art techniques.
</details>
<details>
<summary><u><strong>Multimodal Fusion of EHR in Structures and Semantics: Integrating Clinical Records and Notes with Hypergraph and LLM (2024)</strong></u></summary>

**Authors:** Hejie Cui, Xinyu Fang, Ran Xu, Xuan Kan, Joyce C. Ho, Carl Yang  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2403.08818](https://arxiv.org/abs/2403.08818)  
**Abstract:**  
Electronic Health Records (EHRs) have become increasingly popular to support clinical decision-making and healthcare in recent decades. EHRs usually contain heterogeneous information, such as structural data in tabular form and unstructured data in textual notes. Different types of information in EHRs can complement each other and provide a more complete picture of the health status of a patient. While there has been a lot of research on representation learning of structured EHR data, the fusion of different types of EHR data (multimodal fusion) is not well studied. This is mostly because of the complex medical coding systems used and the noise and redundancy present in the written notes. In this work, we propose a new framework called MINGLE, which integrates both structures and semantics in EHR effectively. Our framework uses a two-level infusion strategy to combine medical concept semantics and clinical note semantics into hypergraph neural networks, which learn the complex interactions between different types of data to generate visit representations for downstream prediction. Experiment results on two EHR datasets, the public MIMIC-III and private CRADLE, show that MINGLE can effectively improve predictive performance by 11.83% relatively, enhancing semantic integration as well as multimodal fusion for structural and textual EHR data.
</details>
<details>
<summary><u><strong>One for All: Towards Training One Graph Model for All Classification Tasks (2024)</strong></u></summary>

**Authors:** Hao Liu, Jiarui Feng, Lecheng Kong, Ningyue Liang, Dacheng Tao, Yixin Chen, Muhan Zhang  
**Venue & Year:** International Conference on Learning Representations, 2024  
**Link:** [https://openreview.net/forum?id=4IT2pgc9v6](https://openreview.net/forum?id=4IT2pgc9v6)  
**Abstract:**  
Designing a single model to address multiple tasks has been a long-standing objective in artificial intelligence. Recently, large language models have demonstrated exceptional capability in solving different tasks within the language domain. However, a unified model for various graph tasks remains underexplored, primarily due to the challenges unique to the graph learning domain. First, graph data from different areas carry distinct attributes and follow different distributions. Such discrepancy makes it hard to represent graphs in a single representation space. Second, tasks on graphs diversify into node, link, and graph tasks, requiring distinct embedding strategies. Finally, an appropriate graph prompting paradigm for in-context learning is unclear. We propose One for All (OFA), the first general framework that can use a single graph model to address the above challenges. Specifically, OFA proposes text-attributed graphs to unify different graph data by describing nodes and edges with natural language and uses language models to encode the diverse and possibly cross-domain text attributes to feature vectors in the same embedding space. Furthermore, OFA introduces the concept of nodes-of-interest to standardize different tasks with a single task representation. For in-context learning on graphs, OFA introduces a novel graph prompting paradigm that appends prompting substructures to the input graph, which enables it to address varied tasks without fine-tuning. We train the OFA model using graph data from multiple domains (including citation networks, molecular graphs, knowledge graphs, etc.) simultaneously and evaluate its ability in supervised, few-shot, and zero-shot learning scenarios. OFA performs well across different tasks, making it the first general-purpose across-domains classification model on graphs.
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
**Abstract:**  
Existing foundation models, such as CLIP, aim to learn a unified embedding space for multimodal data, enabling a wide range of downstream web-based applications like search, recommendation, and content classification. However, these models often overlook the inherent graph structures in multimodal datasets, where entities and their relationships are crucial. Multimodal graphs (MMGs) represent such graphs where each node is associated with features from different modalities, while the edges capture the relationships between these entities. On the other hand, existing graph foundation models primarily focus on text-attributed graphs (TAGs) and are not designed to handle the complexities of MMGs. To address these limitations, we propose UniGraph2, a novel cross-domain graph foundation model that enables general representation learning on MMGs, providing a unified embedding space. UniGraph2 employs modality-specific encoders alongside a graph neural network (GNN) to learn a unified low-dimensional embedding space that captures both the multimodal information and the underlying graph structure. We propose a new cross-domain multi-graph pre-training algorithm at scale to ensure effective transfer learning across diverse graph domains and modalities. Additionally, we adopt a Mixture of Experts (MoE) component to align features from different domains and modalities, ensuring coherent and robust embeddings that unify the information across modalities. Extensive experiments on a variety of multimodal graph tasks demonstrate that UniGraph2 significantly outperforms state-of-the-art models in tasks such as representation learning, transfer learning, and multimodal generative tasks, offering a scalable and flexible solution for learning on MMGs.
</details>
<details>
<summary><u><strong>LLMRec: Large language models with graph augmentation for recommendation (2024)</strong></u></summary>

**Authors:** Wei Wei, Xubin Ren, Jiabin Tang, Qinyong Wang, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, Chao Huang  
**Venue & Year:** Proceedings of the 17th ACM International Conference on Web Search and Data Mining, 2024  
**Link:** [https://doi.org/10.1145/3616855.3635853](https://doi.org/10.1145/3616855.3635853)  
**Abstract:**  
The problem of data sparsity has long been a challenge in recommendation systems, and previous studies have attempted to address this issue by incorporating side information. However, this approach often introduces side effects such as noise, availability issues, and low data quality, which in turn hinder the accurate modeling of user preferences and adversely impact recommendation performance. In light of the recent advancements in large language models (LLMs), which possess extensive knowledge bases and strong reasoning capabilities, we propose a novel framework called LLMRec that enhances recommender systems by employing three simple yet effective LLM-based graph augmentation strategies. Our approach leverages the rich content available within online platforms (e.g., Netflix, MovieLens) to augment the interaction graph in three ways: (i) reinforcing user-item interaction egde, (ii) enhancing the understanding of item node attributes, and (iii) conducting user node profiling, intuitively from the natural language perspective. By employing these strategies, we address the challenges posed by sparse implicit feedback and low-quality side information in recommenders. Besides, to ensure the quality of the augmentation, we develop a denoised data robustification mechanism that includes techniques of noisy implicit feedback pruning and MAE-based feature enhancement that help refine the augmented data and improve its reliability. Furthermore, we provide theoretical analysis to support the effectiveness of LLMRec and clarify the benefits of our method in facilitating model optimization. Experimental results on benchmark datasets demonstrate the superiority of our LLM-based augmentation approach over state-of-the-art techniques.
</details>
<details>
<summary><u><strong>Touchup-G: Improving feature representation through graph-centric finetuning (2024)</strong></u></summary>

**Authors:** Jing Zhu, Xiang Song, Vassilis Ioannidis, Danai Koutra, Christos Faloutsos  
**Venue & Year:** Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2024  
**Link:** [https://doi.org/10.1145/3626772.3657978](https://doi.org/10.1145/3626772.3657978)  
**Abstract:**  
How can we enhance the node features acquired from Pretrained Models (PMs) to better suit downstream graph learning tasks? Graph Neural Networks (GNNs) have become the state-of-the-art approach for many high-impact, real-world graph applications. For feature-rich graphs, a prevalent practice involves directly utilizing a PM to generate features. Nevertheless, this practice is suboptimal as the node features extracted from PMs are graph-agnostic and prevent GNNs from fully utilizing the potential correlations between the graph structure and node features, leading to a decline in GNN performance. In this work, we seek to improve the node features obtained from a PM for graph tasks and introduce TouchUp-G, a "Detect & Correct" approach for refining node features extracted from PMs. TouchUp-G detects the alignment using a novel feature homophily metric and corrects the misalignment through a simple touchup on the PM. It is (a) General: applicable to any downstream graph task; (b) Multi-modal: able to improve raw features of any modality; (c) Principled: it is closely related to a novel metric, feature homophily, which we propose to quantify the alignment between the graph structure and node features; (d) Effective: achieving state-of-the-art results on four real-world datasets spanning different tasks and modalities.
</details>
<details>
<summary><u><strong>GraphAdapter: Tuning vision-language models with dual knowledge graph (2024)</strong></u></summary>

**Authors:** Xin Li, Dongze Lian, Zhihe Lu, Jiawang Bai, Zhibo Chen, Xinchao Wang  
**Venue & Year:** Advances in Neural Information Processing Systems, 2024  
**Link:** [https://openreview.net/forum?id=YmEDnMynuO&noteId=0rFYtJNqHc](https://openreview.net/forum?id=YmEDnMynuO&noteId=0rFYtJNqHc)  
**Abstract:**  
Adapter-style efficient transfer learning (ETL) has shown excellent performance in the tuning of vision-language models (VLMs) under the low-data regime, where only a few additional parameters are introduced to excavate the task-specific knowledge based on the general and powerful representation of VLMs. However, most adapter-style works face two limitations: (i) modeling task-specific knowledge with a single modality only; and (ii) overlooking the exploitation of the inter-class relationships in downstream tasks, thereby leading to sub-optimal solutions. To mitigate that, we propose an effective adapter-style tuning strategy, dubbed GraphAdapter, which performs the textual adapter by explicitly modeling the dual-modality structure knowledge (i.e., the correlation of different semantics/classes in textual and visual modalities) with a dual knowledge graph. In particular, the dual knowledge graph is established with two sub-graphs, i.e., a textual knowledge sub-graph, and a visual knowledge sub-graph, where the nodes and edges represent the semantics/classes and their correlations in two modalities, respectively. This enables the textual feature of each prompt to leverage the task-specific structure knowledge from both textual and visual modalities, yielding a more effective classifier for downstream tasks. Extensive experimental results on 11 benchmark datasets reveal that our GraphAdapter significantly outperforms the previous adapter-based methods.
</details>
<details>
<summary><u><strong>When Graph meets Multimodal: Benchmarking and Meditating on Multimodal Attributed Graphs Learning (2024)</strong></u></summary>

**Authors:** Hao Yan, Chaozhuo Li, Jun Yin, Zhigang Yu, Weihao Han, Mingzheng Li, Zhengxin Zeng, Hao Sun, Senzhang Wang  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2410.09132](https://arxiv.org/abs/2410.09132)  
**Abstract:**  
Multimodal Attributed Graphs (MAGs) are ubiquitous in real-world applications, encompassing extensive knowledge through multimodal attributes attached to nodes (e.g., texts and images) and topological structure representing node interactions. Despite its potential to advance diverse research fields like social networks and e-commerce, MAG representation learning (MAGRL) remains underexplored due to the lack of standardized datasets and evaluation frameworks. In this paper, we first propose MAGB, a comprehensive MAG benchmark dataset, featuring curated graphs from various domains with both textual and visual attributes. Based on MAGB dataset, we further systematically evaluate two mainstream MAGRL paradigms: GNN-as-Predictor, which integrates multimodal attributes via Graph Neural Networks (GNNs), and VLM-as-Predictor, which harnesses Vision Language Models (VLMs) for zero-shot reasoning. Extensive experiments on MAGB reveal following critical insights: (i) Modality significances fluctuate drastically with specific domain characteristics. (ii) Multimodal embeddings can elevate the performance ceiling of GNNs. However, intrinsic biases among modalities may impede effective training, particularly in low-data scenarios. (iii) VLMs are highly effective at generating multimodal embeddings that alleviate the imbalance between textual and visual attributes. These discoveries, which illuminate the synergy between multimodal attributes and graph topologies, contribute to reliable benchmarks, paving the way for future MAG research.
</details>
<details>
<summary><u><strong>Multimodal Graph Learning for Generative Tasks (2023)</strong></u></summary>

**Authors:** Minji Yoon, Jing Yu Koh, Bryan Hooi, Russ Salakhutdinov  
**Venue & Year:** NeurIPS 2023 Workshop: New Frontiers in Graph Learning, 2023  
**Link:** [https://openreview.net/forum?id=YILik4gFBk](https://openreview.net/forum?id=YILik4gFBk)  
**Abstract:**  
Multimodal learning combines multiple data modalities, broadening the types and complexity of data our models can utilize; for example, from plain text to image-caption pairs. Most multimodal learning algorithms focus on modeling simple one-to-one pairs of data from two modalities, such as image-caption pairs, or audio-text pairs. However, in most real-world settings, entities of different modalities interact with each other in more complex and multifaceted ways, going beyond one-to-one mappings. We propose to represent these complex relationships as graphs, allowing us to capture data with any number of modalities, and with complex relationships between modalities that can flexibly vary from one sample to another. Toward this goal, we propose Multimodal Graph Learning (MMGL), a general and systematic framework for capturing information from multiple multimodal neighbors with relational structures among them. In particular, we focus on MMGL for \emph{generative} tasks, building upon pretrained Language Models (LMs), aiming to augment their text generation with multimodal neighbor contexts. We study three research questions raised by MMGL: (1) how can we infuse multiple neighbor information into the pretrained LMs, while avoiding scalability issues? (2) how can we infuse the graph structure information among multimodal neighbors into the LMs? and (3) how can we finetune the pretrained LMs to learn from the neighbor context parameter-efficiently? We conduct extensive experiments to answer these three questions on MMGL and analyze the empirical results to pave the way for future MMGL research.
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
**Abstract:**  
Existing benchmarks like NLGraph and GraphQA evaluate LLMs on graphs by focusing mainly on pairwise relationships, overlooking the high-order correlations found in real-world data. Hypergraphs, which can model complex beyond-pairwise relationships, offer a more robust framework but are still underexplored in the context of LLMs. To address this gap, we introduce LLM4Hypergraph, the first comprehensive benchmark comprising 21,500 problems across eight low-order, five high-order, and two isomorphism tasks, utilizing both synthetic and real-world hypergraphs from citation networks and protein structures. We evaluate six prominent LLMs, including GPT-4o, demonstrating our benchmark’s effectiveness in identifying model strengths and weaknesses. Our specialized prompt- ing framework incorporates seven hypergraph languages and introduces two novel techniques, Hyper-BAG and Hyper-COT, which enhance high-order reasoning and achieve an average 4% (up to 9%) performance improvement on structure classification tasks. This work establishes a foundational testbed for integrating hypergraph computational capabilities into LLMs, advancing their comprehension.
</details>
<details>
<summary><u><strong>UniGraph2: Learning a Unified Embedding Space to Bind Multimodal Graphs (2025)</strong></u></summary>

**Authors:** Yufei He, Yuan Sui, Xiaoxin He, Yue Liu, Yifei Sun, Bryan Hooi  
**Venue & Year:** arXiv preprint, 2025  
**Link:** [https://arxiv.org/abs/2502.00806](https://arxiv.org/abs/2502.00806)  
**Abstract:**  
Existing foundation models, such as CLIP, aim to learn a unified embedding space for multimodal data, enabling a wide range of downstream web-based applications like search, recommendation, and content classification. However, these models often overlook the inherent graph structures in multimodal datasets, where entities and their relationships are crucial. Multimodal graphs (MMGs) represent such graphs where each node is associated with features from different modalities, while the edges capture the relationships between these entities. On the other hand, existing graph foundation models primarily focus on text-attributed graphs (TAGs) and are not designed to handle the complexities of MMGs. To address these limitations, we propose UniGraph2, a novel cross-domain graph foundation model that enables general representation learning on MMGs, providing a unified embedding space. UniGraph2 employs modality-specific encoders alongside a graph neural network (GNN) to learn a unified low-dimensional embedding space that captures both the multimodal information and the underlying graph structure. We propose a new cross-domain multi-graph pre-training algorithm at scale to ensure effective transfer learning across diverse graph domains and modalities. Additionally, we adopt a Mixture of Experts (MoE) component to align features from different domains and modalities, ensuring coherent and robust embeddings that unify the information across modalities. Extensive experiments on a variety of multimodal graph tasks demonstrate that UniGraph2 significantly outperforms state-of-the-art models in tasks such as representation learning, transfer learning, and multimodal generative tasks, offering a scalable and flexible solution for learning on MMGs.
</details>
<details>
<summary><u><strong>Path-LLM: A Shortest-Path-based LLM Learning for Unified Graph Representation (2024)</strong></u></summary>

**Authors:** Wenbo Shang, Xuliang Zhu, Xin Huang  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2408.05456](https://arxiv.org/abs/2408.05456)  
**Abstract:**  
Unified graph representation learning aims to generate node embeddings, which can be applied to multiple downstream applications of graph analytics. However, existing studies based on graph neural networks and language models either suffer from the limitations of numerous training needs toward specific downstream predictions, poor generalization, or shallow semantic features. In this work, we propose a novel Path-LLM model to efficiently learn unified graph representation, which leverages a powerful large language model (LLM) to incorporate our proposed path features. Our Path-LLM framework consists of four well-designed techniques. First, we develop a new mechanism of long-to-short shortest path (L2SP) selection, which can cover key connections between different dense groups. An in-depth analysis and comparison of different path selections is conducted to justify the rationale behind our designed L2SP method. Next, we design path textualization to obtain L2SP-based training texts with key phrase selection from node text attributes. We then feed the texts into a self-supervised LLM training process to align next node/edge generation in L2SP with next token generation in causal language modeling for graph representation learning and finally extract the unified graph embeddings. We theoretically analyze the algorithm complexity of our Path-LLM approach. Extensive experiments on large-scale graph benchmarks validate the superiority of Path-LLM against state-of-the-art methods WalkLM, GraphGPT, OFA, and GraphTranslator on two classical graph learning tasks (node classification and edge validation) and one NP-hard graph query processing task (keyword search). Compared with WalkLM, our approach saves more than 90% of training paths on millions-scale graphs and runs at most 35x faster.
</details>
<details>
<summary><u><strong>Graphadapter: Tuning vision-language models with dual knowledge graph (2024)</strong></u></summary>

**Authors:** Xin Li, Dongze Lian, Zhihe Lu, Jiawang Bai, Zhibo Chen, Xinchao Wang  
**Venue & Year:** Advances in Neural Information Processing Systems, 2024  
**Link:** [https://openreview.net/forum?id=YmEDnMynuO&noteId=0rFYtJNqHc](https://openreview.net/forum?id=YmEDnMynuO&noteId=0rFYtJNqHc)  
**Abstract:**  
Adapter-style efficient transfer learning (ETL) has shown excellent performance in the tuning of vision-language models (VLMs) under the low-data regime, where only a few additional parameters are introduced to excavate the task-specific knowledge based on the general and powerful representation of VLMs. However, most adapter-style works face two limitations: (i) modeling task-specific knowledge with a single modality only; and (ii) overlooking the exploitation of the inter-class relationships in downstream tasks, thereby leading to sub-optimal solutions. To mitigate that, we propose an effective adapter-style tuning strategy, dubbed GraphAdapter, which performs the textual adapter by explicitly modeling the dual-modality structure knowledge (i.e., the correlation of different semantics/classes in textual and visual modalities) with a dual knowledge graph. In particular, the dual knowledge graph is established with two sub-graphs, i.e., a textual knowledge sub-graph, and a visual knowledge sub-graph, where the nodes and edges represent the semantics/classes and their correlations in two modalities, respectively. This enables the textual feature of each prompt to leverage the task-specific structure knowledge from both textual and visual modalities, yielding a more effective classifier for downstream tasks. Extensive experimental results on 11 benchmark datasets reveal that our GraphAdapter significantly outperforms the previous adapter-based methods.
</details>
<details>
<summary><u><strong>LLMRec: Large language models with graph augmentation for recommendation (2024)</strong></u></summary>

**Authors:** Wei Wei, Xubin Ren, Jiabin Tang, Qinyong Wang, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, Chao Huang  
**Venue & Year:** Proceedings of the 17th ACM International Conference on Web Search and Data Mining, 2024  
**Link:** [https://doi.org/10.1145/3616855.3635853](https://doi.org/10.1145/3616855.3635853)  
**Abstract:**  
The problem of data sparsity has long been a challenge in recommendation systems, and previous studies have attempted to address this issue by incorporating side information. However, this approach often introduces side effects such as noise, availability issues, and low data quality, which in turn hinder the accurate modeling of user preferences and adversely impact recommendation performance. In light of the recent advancements in large language models (LLMs), which possess extensive knowledge bases and strong reasoning capabilities, we propose a novel framework called LLMRec that enhances recommender systems by employing three simple yet effective LLM-based graph augmentation strategies. Our approach leverages the rich content available within online platforms (e.g., Netflix, MovieLens) to augment the interaction graph in three ways: (i) reinforcing user-item interaction egde, (ii) enhancing the understanding of item node attributes, and (iii) conducting user node profiling, intuitively from the natural language perspective. By employing these strategies, we address the challenges posed by sparse implicit feedback and low-quality side information in recommenders. Besides, to ensure the quality of the augmentation, we develop a denoised data robustification mechanism that includes techniques of noisy implicit feedback pruning and MAE-based feature enhancement that help refine the augmented data and improve its reliability. Furthermore, we provide theoretical analysis to support the effectiveness of LLMRec and clarify the benefits of our method in facilitating model optimization. Experimental results on benchmark datasets demonstrate the superiority of our LLM-based augmentation approach over state-of-the-art techniques.
</details>
<details>
<summary><u><strong>Talk like a Graph: Encoding Graphs for Large Language Models (2024)</strong></u></summary>

**Authors:** Bahare Fatemi, Jonathan Halcrow, Bryan Perozzi  
**Venue & Year:** International Conference on Learning Representations, 2024  
**Link:** [https://openreview.net/forum?id=IuXR1CCrSi](https://openreview.net/forum?id=IuXR1CCrSi)  
**Abstract:**  
Graphs are a powerful tool for representing and analyzing complex relationships in real-world applications such as social networks, recommender systems, and computational finance. Reasoning on graphs is essential for drawing inferences about the relationships between entities in a complex system, and to identify hidden patterns and trends. Despite the remarkable progress in automated reasoning with natural text, reasoning on graphs with large language models (LLMs) remains an understudied problem. In this work, we perform the first comprehensive study of encoding graph-structured data as text for consumption by LLMs. We show that LLM performance on graph reasoning tasks varies on three fundamental levels: (1) the graph encoding method, (2) the nature of the graph task itself, and (3) interestingly, the very structure of the graph considered. These novel results provide valuable insight on strategies for encoding graphs as text. Using these insights we illustrate how the correct choice of encoders can boost performance on graph reasoning tasks inside LLMs by 4.8% to 61.8%, depending on the task.
</details>
<details>
<summary><u><strong>Gita: Graph to visual and textual integration for vision-language graph reasoning (2024)</strong></u></summary>

**Authors:** Yanbin Wei, Shuai Fu, Weisen Jiang, Zejian Zhang, Zhixiong Zeng, Qi Wu, James Kwok, Yu Zhang  
**Venue & Year:** Advances in Neural Information Processing Systems, 2024  
**Link:** [https://openreview.net/forum?id=SaodQ13jga](https://openreview.net/forum?id=SaodQ13jga)  
**Abstract:**  
Large Language Models (LLMs) are increasingly used for various tasks with graph structures. Though LLMs can process graph information in a textual format, they overlook the rich vision modality, which is an intuitive way for humans to comprehend structural information and conduct general graph reasoning. The potential benefits and capabilities of representing graph structures as visual images (i.e., visual graph) are still unexplored. To fill the gap, we innovatively propose an end-to-end framework, called Graph to vIsual and Textual IntegrAtion (GITA), which firstly incorporates visual graphs into general graph reasoning. Besides, we establish Graph-based Vision-Language Question Answering (GVLQA) dataset from existing graph data, which is the first vision-language dataset for general graph reasoning purposes. Extensive experiments on the GVLQA dataset and five real-world datasets show that GITA outperforms mainstream LLMs in terms of general graph reasoning capabilities. Moreover, We highlight the effectiveness of the layout augmentation on visual graphs and pretraining on the GVLQA dataset.
</details>
<details>
<summary><u><strong>WalkLM: A uniform language model fine-tuning framework for attributed graph embedding (2024)</strong></u></summary>

**Authors:** Yanchao Tan, Zihao Zhou, Hang Lv, Weiming Liu, Carl Yang  
**Venue & Year:** Advances in Neural Information Processing Systems, 2024  
**Link:** [https://openreview.net/forum?id=ZrG8kTbt70](https://openreview.net/forum?id=ZrG8kTbt70)  
**Abstract:**  
Graphs are widely used to model interconnected entities and improve downstream predictions in various real-world applications. However, real-world graphs nowadays are often associated with complex attributes on multiple types of nodes and even links that are hard to model uniformly, while the widely used graph neural networks (GNNs) often require sufficient training toward specific downstream predictions to achieve strong performance. In this work, we take a fundamentally different approach than GNNs, to simultaneously achieve deep joint modeling of complex attributes and flexible structures of real-world graphs and obtain unsupervised generic graph representations that are not limited to specific downstream predictions. Our framework, built on a natural integration of language models (LMs) and random walks (RWs), is straightforward, powerful and data-efficient. Specifically, we first perform attributed RWs on the graph and design an automated program to compose roughly meaningful textual sequences directly from the attributed RWs; then we fine-tune an LM using the RW-based textual sequences and extract embedding vectors from the LM, which encapsulates both attribute semantics and graph structures. In our experiments, we evaluate the learned node embeddings towards different downstream prediction tasks on multiple real-world attributed graph datasets and observe significant improvements over a comprehensive set of state-of-the-art unsupervised node embedding methods. We believe this work opens a door for more sophisticated technical designs and empirical evaluations toward the leverage of LMs for the modeling of real-world graphs.
</details>
<details>
<summary><u><strong>Language is all a graph needs (2024)</strong></u></summary>

**Authors:** Ruosong Ye, Caiqi Zhang, Runhui Wang, Shuyuan Xu, Yongfeng Zhang  
**Venue & Year:** Findings of the Association for Computational Linguistics: EACL, 2024  
**Link:** [https://aclanthology.org/2024.findings-eacl.132/](https://aclanthology.org/2024.findings-eacl.132/)  
**Abstract:**  
The emergence of large-scale pre-trained language models has revolutionized various AI research domains. Transformers-based Large Language Models (LLMs) have gradually replaced CNNs and RNNs to unify fields of computer vision and natural language processing. Compared with independent data like images, videos or texts, graphs usually contain rich structural and relational information. Meanwhile, languages, especially natural language, being one of the most expressive mediums, excels in describing complex structures. However, existing work on incorporating graph problems into the generative language modeling framework remains very limited. Considering the rising prominence of LLMs, it becomes essential to explore whether LLMs can also replace GNNs as the foundation model for graphs. In this paper, we propose InstructGLM (Instruction-finetuned Graph Language Model) with highly scalable prompts based on natural language instructions. We use natural language to describe multi-scale geometric structure of the graph and then instruction finetune an LLM to perform graph tasks, which enables Generative Graph Learning. Our method surpasses all GNN baselines on ogbn-arxiv, Cora and PubMed datasets, underscoring its effectiveness and sheds light on generative LLMs as new foundation model for graph machine learning.
</details>
<details>
<summary><u><strong>Graph neural prompting with large language models (2024)</strong></u></summary>

**Authors:** Yijun Tian, Huan Song, Zichen Wang, Haozhu Wang, Ziqing Hu, Fang Wang, Nitesh V Chawla, Panpan Xu  
**Venue & Year:** Proceedings of the AAAI Conference on Artificial Intelligence, 2024  
**Link:** [https://doi.org/10.1609/aaai.v38i17.29875](https://doi.org/10.1609/aaai.v38i17.29875)  
**Abstract:**  
Large language models (LLMs) have shown remarkable generalization capability with exceptional performance in various language modeling tasks. However, they still exhibit inherent limitations in precisely capturing and returning grounded knowledge. While existing work has explored utilizing knowledge graphs (KGs) to enhance language modeling via joint training and customized model architectures, applying this to LLMs is problematic owing to their large number of parameters and high computational cost. Therefore, how to enhance pre-trained LLMs using grounded knowledge, e.g., retrieval-augmented generation, remains an open question. In this work, we propose Graph Neural Prompting (GNP), a novel plug-and-play method to assist pre-trained LLMs in learning beneficial knowledge from KGs. GNP encompasses various designs, including a standard graph neural network encoder, a cross-modality pooling module, a domain projector, and a self-supervised link prediction objective. Extensive experiments on multiple datasets demonstrate the superiority of GNP on both commonsense and biomedical reasoning tasks across different LLM sizes and settings.
</details>
<details>
<summary><u><strong>Let your graph do the talking: Encoding structured data for LLMs (2024)</strong></u></summary>

**Authors:** Bryan Perozzi, Bahare Fatemi, Dustin Zelle, Anton Tsitsulin, Mehran Kazemi, Rami Al-Rfou, Jonathan Halcrow  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2402.05862](https://arxiv.org/abs/2402.05862)  
**Abstract:**  
How can we best encode structured data into sequential form for use in large language models (LLMs)? In this work, we introduce a parameter-efficient method to explicitly represent structured data for LLMs. Our method, GraphToken, learns an encoding function to extend prompts with explicit structured information. Unlike other work which focuses on limited domains (e.g. knowledge graph representation), our work is the first effort focused on the general encoding of structured data to be used for various reasoning tasks. We show that explicitly representing the graph structure allows significant improvements to graph reasoning tasks. Specifically, we see across the board improvements - up to 73% points - on node, edge and, graph-level tasks from the GraphQA benchmark.
</details>
<details>
<summary><u><strong>GraphGPT: Graph instruction tuning for large language models (2024)</strong></u></summary>

**Authors:** Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Suqi Cheng, Dawei Yin, Chao Huang  
**Venue & Year:** Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2024  
**Link:** [https://doi.org/10.1145/3626772.3657775](https://doi.org/10.1145/3626772.3657775)  
**Abstract:**  
Graph Neural Networks (GNNs) have evolved to understand graph structures through recursive exchanges and aggregations among nodes. To enhance robustness, self-supervised learning (SSL) has become a vital tool for data augmentation. Traditional methods often depend on fine-tuning with task-specific labels, limiting their effectiveness when labeled data is scarce. Our research tackles this by advancing graph model generalization in zero-shot learning environments. Inspired by the success of large language models (LLMs), we aim to create a graph-oriented LLM capable of exceptional generalization across various datasets and tasks without relying on downstream graph data. We introduce the GraphGPT framework, which integrates LLMs with graph structural knowledge through graph instruction tuning. This framework includes a text-graph grounding component to link textual and graph structures and a dual-stage instruction tuning approach with a lightweight graph-text alignment projector. These innovations allow LLMs to comprehend complex graph structures and enhance adaptability across diverse datasets and tasks. Our framework demonstrates superior generalization in both supervised and zero-shot graph learning tasks, surpassing existing benchmarks.
</details>
<details>
<summary><u><strong>Higpt: Heterogeneous graph language model (2024)</strong></u></summary>

**Authors:** Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Long Xia, Dawei Yin, Chao Huang  
**Venue & Year:** Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024  
**Link:** [https://doi.org/10.1145/3637528.3671987](https://doi.org/10.1145/3637528.3671987)  
**Abstract:**  
Heterogeneous graph learning aims to capture complex relationships and diverse relational semantics among entities in a heterogeneous graph to obtain meaningful representations for nodes and edges. Recent advancements in heterogeneous graph neural networks (HGNNs) have achieved state-of-the-art performance by considering relation heterogeneity and using specialized message functions and aggregation rules. However, existing frameworks for heterogeneous graph learning have limitations in generalizing across diverse heterogeneous graph datasets. Most of these frameworks follow the "pre-train" and "fine-tune" paradigm on the same dataset, which restricts their capacity to adapt to new and unseen data. This raises the question: "Can we generalize heterogeneous graph models to be well-adapted to diverse downstream learning tasks with distribution shifts in both node token sets and relation type heterogeneity?" To tackle those challenges, we propose HiGPT, a general large graph model with Heterogeneous graph instruction-tuning paradigm. Our framework enables learning from arbitrary heterogeneous graphs without the need for any fine-tuning process from downstream datasets. To handle distribution shifts in heterogeneity, we introduce an in-context heterogeneous graph tokenizer that captures semantic relationships in different heterogeneous graphs, facilitating model adaptation. We incorporate a large corpus of heterogeneity-aware graph instructions into our HiGPT, enabling the model to effectively comprehend complex relation heterogeneity and distinguish between various types of graph tokens. Furthermore, we introduce the Mixture-of-Thought (MoT) instruction augmentation paradigm to mitigate data scarcity by generating diverse and informative instructions. Through comprehensive evaluations conducted in various settings, our proposed framework demonstrates exceptional performance in terms of generalization performance, surpassing current leading benchmarks.
</details>
<details>
<summary><u><strong>GPT4Graph: Can large language models understand graph structured data? An empirical evaluation and benchmarking (2023)</strong></u></summary>

**Authors:** Jiayan Guo, Lun Du, Hengyu Liu, Mengyu Zhou, Xinyi He, Shi Han  
**Venue & Year:** arXiv preprint, 2023  
**Link:** [https://arxiv.org/abs/2305.15066](https://arxiv.org/abs/2305.15066)  
**Abstract:**  
Large language models~(LLM) like ChatGPT have become indispensable to artificial general intelligence~(AGI), demonstrating excellent performance in various natural language processing tasks. In the real world, graph data is ubiquitous and an essential part of AGI and prevails in domains like social network analysis, bioinformatics and recommender systems. The training corpus of large language models often includes some algorithmic components, which allows them to achieve certain effects on some graph data-related problems. However, there is still little research on their performance on a broader range of graph-structured data. In this study, we conduct an extensive investigation to assess the proficiency of LLMs in comprehending graph data, employing a diverse range of structural and semantic-related tasks. Our analysis encompasses 10 distinct tasks that evaluate the LLMs' capabilities in graph understanding. Through our study, we not only uncover the current limitations of language models in comprehending graph structures and performing associated reasoning tasks but also emphasize the necessity for further advancements and novel approaches to enhance their graph processing capabilities. Our findings contribute valuable insights towards bridging the gap between language models and graph understanding, paving the way for more effective graph mining and knowledge extraction.
</details>
<details>
<summary><u><strong>Graphtext: Graph reasoning in text space (2023)</strong></u></summary>

**Authors:** Jianan Zhao, Le Zhuo, Yikang Shen, Meng Qu, Kai Liu, Michael Bronstein, Zhaocheng Zhu, Jian Tang  
**Venue & Year:** arXiv preprint, 2023  
**Link:** [https://arxiv.org/abs/2310.01089](https://arxiv.org/abs/2310.01089)  
**Abstract:**  
Large Language Models (LLMs) have gained the ability to assimilate human knowledge and facilitate natural language interactions with both humans and other LLMs. However, despite their impressive achievements, LLMs have not made significant advancements in the realm of graph machine learning. This limitation arises because graphs encapsulate distinct relational data, making it challenging to transform them into natural language that LLMs understand. In this paper, we bridge this gap with a novel framework, GraphText, that translates graphs into natural language. GraphText derives a graph-syntax tree for each graph that encapsulates both the node attributes and inter-node relationships. Traversal of the tree yields a graph text sequence, which is then processed by an LLM to treat graph tasks as text generation tasks. Notably, GraphText offers multiple advantages. It introduces training-free graph reasoning: even without training on graph data, GraphText with ChatGPT can achieve on par with, or even surpassing, the performance of supervised-trained graph neural networks through in-context learning (ICL). Furthermore, GraphText paves the way for interactive graph reasoning, allowing both humans and LLMs to communicate with the model seamlessly using natural language. These capabilities underscore the vast, yet-to-be-explored potential of LLMs in the domain of graph machine learning.
</details>
<details>
<summary><u><strong>Can language models solve graph problems in natural language? (2023)</strong></u></summary>

**Authors:** Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, Yulia Tsvetkov  
**Venue & Year:** Advances in Neural Information Processing Systems, 2023  
**Link:** [https://openreview.net/forum?id=UDqHhbqYJV](https://openreview.net/forum?id=UDqHhbqYJV)  
**Abstract:**  
Large language models (LLMs) are increasingly adopted for a variety of tasks with implicit graphical structures, such as planning in robotics, multi-hop question answering or knowledge probing, structured commonsense reasoning, and more. While LLMs have advanced the state-of-the-art on these tasks with structure implications, whether LLMs could explicitly process textual descriptions of graphs and structures, map them to grounded conceptual spaces, and perform structured operations remains underexplored. To this end, we propose NLGraph (Natural Language Graph), a comprehensive benchmark of graph-based problem solving designed in natural language. NLGraph contains 29,370 problems, covering eight graph reasoning tasks with varying complexity from simple tasks such as connectivity and shortest path up to complex problems such as maximum flow and simulating graph neural networks. We evaluate LLMs (GPT-3/4) with various prompting approaches on the NLGraph benchmark and find that 1) language models do demonstrate preliminary graph reasoning abilities, 2) the benefit of advanced prompting and in-context learning diminishes on more complex graph problems, while 3) LLMs are also (un)surprisingly brittle in the face of spurious correlations in graph and problem settings. We then propose Build-a-Graph Prompting and Algorithmic Prompting, two instruction-based approaches to enhance LLMs in solving natural language graph problems. Build-a-Graph and Algorithmic prompting improve the performance of LLMs on NLGraph by 3.07% to 16.85% across multiple tasks and settings, while how to solve the most complicated graph reasoning tasks in our setup with language models remains an open research question.
</details>
<details>
<summary><u><strong>Evaluating large language models on graphs: Performance insights and comparative analysis (2023)</strong></u></summary>

**Authors:** Chang Liu, Bo Wu  
**Venue & Year:** arXiv preprint, 2023  
**Link:** [https://arxiv.org/abs/2308.11224](https://arxiv.org/abs/2308.11224)  
**Abstract:**  
Large Language Models (LLMs) have garnered considerable interest within both academic and industrial. Yet, the application of LLMs to graph data remains under-explored. In this study, we evaluate the capabilities of four LLMs in addressing several analytical problems with graph data. We employ four distinct evaluation metrics: Comprehension, Correctness, Fidelity, and Rectification. Our results show that: 1) LLMs effectively comprehend graph data in natural language and reason with graph topology. 2) GPT models can generate logical and coherent results, outperforming alternatives in correctness. 3) All examined LLMs face challenges in structural reasoning, with techniques like zero-shot chain-of-thought and few-shot prompting showing diminished efficacy. 4) GPT models often produce erroneous answers in multi-answer tasks, raising concerns in fidelity. 5) GPT models exhibit elevated confidence in their outputs, potentially hindering their rectification capacities. Notably, GPT-4 has demonstrated the capacity to rectify responses from GPT-3.5-turbo and its own previous iterations.
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

<details>
<summary><u><strong>LLM4DyG: Can Large Language Models Solve Spatial-Temporal Problems on Dynamic Graphs? (2024)</strong></u></summary>

**Authors:** Zeyang Zhang, Xin Wang, Ziwei Zhang, Haoyang Li, Yijian Qin, Wenwu Zhu  
**Venue & Year:** Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024  
**Link:** [https://doi.org/10.1145/3637528.3671709](https://doi.org/10.1145/3637528.3671709)  
**Abstract:**  
In an era marked by the increasing adoption of Large Language Models (LLMs) for various tasks, there is a growing focus on exploring LLMs' capabilities in handling web data, particularly graph data. Dynamic graphs, which capture temporal network evolution patterns, are ubiquitous in real-world web data. Evaluating LLMs' competence in understanding spatial-temporal information on dynamic graphs is essential for their adoption in web applications, which remains unexplored in the literature. In this paper, we bridge the gap via proposing to evaluate LLMs' spatial-temporal understanding abilities on dynamic graphs, to the best of our knowledge, for the first time. Specifically, we propose the LLM4DyG benchmark, which includes nine specially designed tasks considering the capability evaluation of LLMs from both temporal and spatial dimensions. Then, we conduct extensive experiments to analyze the impacts of different data generators, data statistics, prompting techniques, and LLMs on the model performance. Finally, we propose Disentangled Spatial-Temporal Thoughts (DST2) for LLMs on dynamic graphs to enhance LLMs' spatial-temporal understanding abilities. Our main observations are: 1) LLMs have preliminary spatial-temporal understanding abilities on dynamic graphs, 2) Dynamic graph tasks show increasing difficulties for LLMs as the graph size and density increase, while not sensitive to the time span and data generation mechanism, 3) the proposed DST2 prompting method can help to improve LLMs' spatial-temporal understanding abilities on dynamic graphs for most tasks. The data and codes are publicly available at Github.
</details>
<details>
<summary><u><strong>TimeR$^4$: Time-aware Retrieval-Augmented Large Language Models for Temporal Knowledge Graph Question Answering (2024)</strong></u></summary>

**Authors:** Xinying Qian, Ying Zhang, Yu Zhao, Baohang Zhou, Xuhui Sui, Li Zhang, Kehui Song  
**Venue & Year:** Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Miami, USA  
**Link:** [https://aclanthology.org/2024.emnlp-main.394/](https://aclanthology.org/2024.emnlp-main.394/)  
**Abstract:**  
Temporal Knowledge Graph Question Answering (TKGQA) aims to answer temporal questions using knowledge in Temporal Knowledge Graphs (TKGs). Previous works employ pre-trained TKG embeddings or graph neural networks to incorporate the knowledge of TKGs. However, these methods fail to fully understand the complex semantic information of time constraints in questions.In contrast, Large Language Models (LLMs) have shown exceptional performance in knowledge graph reasoning, unifying both semantic understanding and structural reasoning. To further enhance LLMs’ temporal reasoning ability, this paper aims to integrate relevant temporal knowledge from TKGs into LLMs through a Time-aware Retrieve-Rewrite-Retrieve-Rerank framework, which we named TimeR4.Specifically, to reduce temporal hallucination in LLMs, we propose a retrieve-rewrite module to rewrite questions using background knowledge stored in the TKGs, thereby acquiring explicit time constraints. Then, we implement a retrieve-rerank module aimed at retrieving semantically and temporally relevant facts from the TKGs and reranking them according to the temporal constraints.To achieve this, we fine-tune a retriever using the contrastive time-aware learning framework.Our approach achieves great improvements, with relative gains of 47.8% and 22.5% on two datasets, underscoring its effectiveness in boosting the temporal reasoning abilities of LLMs. Our code is available at https://github.com/qianxinying/TimeR4.
</details>
<details>
<summary><u><strong>Two-stage Generative Question Answering on Temporal Knowledge Graph Using Large Language Models (2024)</strong></u></summary>

**Authors:** Yifu Gao, Linbo Qiao, Zhigang Kan, Zhihua Wen, Yongquan He, Dongsheng Li  
**Venue & Year:** Findings of the Association for Computational Linguistics: ACL 2024, Bangkok, Thailand  
**Link:** [https://aclanthology.org/2024.findings-acl.401/](https://aclanthology.org/2024.findings-acl.401/)  
**Abstract:**  
Temporal knowledge graph question answering (TKGQA) poses a significant challenge task, due to the temporal constraints hidden in questions and the answers sought from dynamic structured knowledge. Although large language models (LLMs) have made considerable progress in their reasoning ability over structured data, their application to the TKGQA task is a relatively unexplored area. This paper first proposes a novel generative temporal knowledge graph question answering framework, GenTKGQA, which guides LLMs to answer temporal questions through two phases: Subgraph Retrieval and Answer Generation. First, we exploit LLM’s intrinsic knowledge to mine temporal constraints and structural links in the questions without extra training, thus narrowing down the subgraph search space in both temporal and structural dimensions. Next, we design virtual knowledge indicators to fuse the graph neural network signals of the subgraph and the text representations of the LLM in a non-shallow way, which helps the open-source LLM deeply understand the temporal order and structural dependencies among the retrieved facts through instruction tuning. Experimental results on two widely used datasets demonstrate the superiority of our model.
</details>
<details>
<summary><u><strong>Unveiling LLMs: The Evolution of Latent Representations in a Dynamic Knowledge Graph (2024)</strong></u></summary>

**Authors:** Marco Bronzini, Carlo Nicolini, Bruno Lepri, Jacopo Staiano, Andrea Passerini  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2404.03623](https://arxiv.org/abs/2404.03623)  
**Abstract:**  
Large Language Models (LLMs) demonstrate an impressive capacity to recall a vast range of factual knowledge. However, understanding their underlying reasoning and internal mechanisms in exploiting this knowledge remains a key research area. This work unveils the factual information an LLM represents internally for sentence-level claim verification. We propose an end-to-end framework to decode factual knowledge embedded in token representations from a vector space to a set of ground predicates, showing its layer-wise evolution using a dynamic knowledge graph. Our framework employs activation patching, a vector-level technique that alters a token representation during inference, to extract encoded knowledge. Accordingly, we neither rely on training nor external models. Using factual and common-sense claims from two claim verification datasets, we showcase interpretability analyses at local and global levels. The local analysis highlights entity centrality in LLM reasoning, from claim-related information and multi-hop reasoning to representation errors causing erroneous evaluation. On the other hand, the global reveals trends in the underlying evolution, such as word-based knowledge evolving into claim-related facts. By interpreting semantics from LLM latent representations and enabling graph-related analyses, this work enhances the understanding of the factual knowledge resolution process.
</details>
<details>
<summary><u><strong>Large Language Models Can Learn Temporal Reasoning (2024)</strong></u></summary>

**Authors:** Siheng Xiong, Ali Payani, Ramana Kompella, Faramarz Fekri  
**Venue & Year:** Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, 2024  
**Link:** [https://aclanthology.org/2024.acl-long.563/](https://aclanthology.org/2024.acl-long.563/)  
**Abstract:**  
While large language models (LLMs) have demonstrated remarkable reasoning capabilities, they are not without their flaws and inaccuracies. Recent studies have introduced various methods to mitigate these limitations. Temporal reasoning (TR), in particular, presents a significant challenge for LLMs due to its reliance on diverse temporal concepts and intricate temporal logic. In this paper, we propose TG-LLM, a novel framework towards language-based TR. Instead of reasoning over the original context, we adopt a latent representation, temporal graph (TG) that enhances the learning of TR. A synthetic dataset (TGQA), which is fully controllable and requires minimal supervision, is constructed for fine-tuning LLMs on this text-to-TG translation task. We confirmed in experiments that the capability of TG translation learned on our dataset can be transferred to other TR tasks and benchmarks. On top of that, we teach LLM to perform deliberate reasoning over the TGs via Chain-of-Thought (CoT) bootstrapping and graph data augmentation. We observed that those strategies, which maintain a balance between usefulness and diversity, bring more reliable CoTs and final results than the vanilla CoT distillation.
</details>
<details>
<summary><u><strong>Chain-of-History Reasoning for Temporal Knowledge Graph Forecasting (2024)</strong></u></summary>

**Authors:** Yuwei Xia, Ding Wang, Qiang Liu, Liang Wang, Shu Wu, Xiao-Yu Zhang  
**Venue & Year:** Findings of the Association for Computational Linguistics: ACL 2024, Bangkok, Thailand  
**Link:** [https://aclanthology.org/2024.findings-acl.955/](https://aclanthology.org/2024.findings-acl.955/)  
**Abstract:**  
Temporal Knowledge Graph (TKG) forecasting aims to predict future facts based on given histories. Most recent graph-based models excel at capturing structural information within TKGs but lack semantic comprehension abilities. Nowadays, with the surge of LLMs, the LLM-based TKG prediction model has emerged. However, the existing LLM-based model exhibits three shortcomings: (1) It only focuses on the first-order history for prediction while ignoring high-order historical information, resulting in the provided information for LLMs being extremely limited. (2) LLMs struggle with optimal reasoning performance under heavy historical information loads. (3) For TKG prediction, the temporal reasoning capability of LLM alone is limited. To address the first two challenges, we propose Chain-of-History (CoH) reasoning which explores high-order histories step-by-step, achieving effective utilization of high-order historical information for LLMs on TKG prediction. To address the third issue, we design CoH as a plug-and-play module to enhance the performance of graph-based models for TKG prediction. Extensive experiments on three datasets and backbones demonstrate the effectiveness of CoH.
</details>
<details>
<summary><u><strong>zrLLM: Zero-Shot Relational Learning on Temporal Knowledge Graphs with Large Language Models (2024)</strong></u></summary>

**Authors:** Zifeng Ding, Heling Cai, Jingpei Wu, Yunpu Ma, Ruotong Liao, Bo Xiong, Volker Tresp  
**Venue & Year:** Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Mexico City, Mexico  
**Link:** [https://aclanthology.org/2024.naacl-long.104/](https://aclanthology.org/2024.naacl-long.104/)  
**Abstract:**  
Modeling evolving knowledge over temporal knowledge graphs (TKGs) has become a heated topic. Various methods have been proposed to forecast links on TKGs. Most of them are embedding-based, where hidden representations are learned to represent knowledge graph (KG) entities and relations based on the observed graph contexts. Although these methods show strong performance on traditional TKG forecasting (TKGF) benchmarks, they face a strong challenge in modeling the unseen zero-shot relations that have no prior graph context. In this paper, we try to mitigate this problem as follows. We first input the text descriptions of KG relations into large language models (LLMs) for generating relation representations, and then introduce them into embedding-based TKGF methods. LLM-empowered representations can capture the semantic information in the relation descriptions. This makes the relations, whether seen or unseen, with similar semantic meanings stay close in the embedding space, enabling TKGF models to recognize zero-shot relations even without any observed graph context. Experimental results show that our approach helps TKGF models to achieve much better performance in forecasting the facts with previously unseen relations, while still maintaining their ability in link forecasting regarding seen relations.
</details>
<details>
<summary><u><strong>Temporal Knowledge Graph Forecasting Without Knowledge Using In-Context Learning (2023)</strong></u></summary>

**Authors:** Dong-Ho Lee, Kian Ahrabian, Woojeong Jin, Fred Morstatter, Jay Pujara  
**Venue & Year:** Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Singapore  
**Link:** [https://aclanthology.org/2023.emnlp-main.36/](https://aclanthology.org/2023.emnlp-main.36/)  
**Abstract:**  
Temporal knowledge graph (TKG) forecasting benchmarks challenge models to predict future facts using knowledge of past facts. In this paper, we develop an approach to use in-context learning (ICL) with large language models (LLMs) for TKG forecasting. Our extensive evaluation compares diverse baselines, including both simple heuristics and state-of-the-art (SOTA) supervised models, against pre-trained LLMs across several popular benchmarks and experimental settings. We observe that naive LLMs perform on par with SOTA models, which employ carefully designed architectures and supervised training for the forecasting task, falling within the (-3.6%, +1.5%) Hits@1 margin relative to the median performance. To better understand the strengths of LLMs for forecasting, we explore different approaches for selecting historical facts, constructing prompts, controlling information propagation, and parsing outputs into a probability distribution. A surprising finding from our experiments is that LLM performance endures (±0.4% Hit@1) even when semantic information is removed by mapping entities/relations to arbitrary numbers, suggesting that prior semantic knowledge is unnecessary; rather, LLMs can leverage the symbolic patterns in the context to achieve such a strong performance. Our analysis also reveals that ICL enables LLMs to learn irregular patterns from the historical context, going beyond frequency and recency biases
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
**Abstract:**  
Dynamic knowledge graphs (DKGs) are popular structures to express different types of connections between objects over time. They can also serve as an efficient mathematical tool to represent information extracted from complex unstructured data sources, such as text or images. Within financial applications, DKGs could be used to detect trends for strategic thematic investing, based on information obtained from financial news articles. In this work, we explore the properties of large language models (LLMs) as dynamic knowledge graph generators, proposing a novel open-source fine-tuned LLM for this purpose, called the Integrated Contextual Knowledge Graph Generator (ICKG). We use ICKG to produce a novel open-source DKG from a corpus of financial news articles, called FinDKG, and we propose an attention-based GNN architecture for analysing it, called KGTransformer. We test the performance of the proposed model on benchmark datasets and FinDKG, demonstrating superior performance on link prediction tasks. Additionally, we evaluate the performance of the KGTransformer on FinDKG for thematic investing, showing it can outperform existing thematic ETFs.
</details>
<details>
<summary><u><strong>GenTKG: Generative Forecasting on Temporal Knowledge Graph with Large Language Models (2024)</strong></u></summary>

**Authors:** Ruotong Liao, Xu Jia, Yangzhe Li, Yunpu Ma, Volker Tresp  
**Venue & Year:** Findings of the Association for Computational Linguistics: NAACL, 2024  
**Link:** [https://aclanthology.org/2024.findings-naacl.268/](https://aclanthology.org/2024.findings-naacl.268/)  
**Abstract:**  
The rapid advancements in large language models (LLMs) have ignited interest in the temporal knowledge graph (tKG) domain, where conventional embedding-based and rule-based methods dominate. The question remains open of whether pre-trained LLMs can understand structured temporal relational data and replace them as the foundation model for temporal relational forecasting. Therefore, we bring temporal knowledge forecasting into the generative setting. However, challenges occur in the huge chasms between complex temporal graph data structure and sequential natural expressions LLMs can handle, and between the enormous data sizes of tKGs and heavy computation costs of finetuning LLMs. To address these challenges, we propose a novel retrieval-augmented generation framework named GenTKG combining a temporal logical rule-based retrieval strategy and few-shot parameter-efficient instruction tuning to solve the above challenges, respectively. Extensive experiments have shown that GenTKG outperforms conventional methods of temporal relational forecasting with low computation resources using extremely limited training data as few as 16 samples. GenTKG also highlights remarkable cross-domain generalizability with outperforming performance on unseen datasets without re-training, and in-domain generalizability regardless of time split in the same dataset. Our work reveals the huge potential of LLMs in the tKG domain and opens a new frontier for generative forecasting on tKGs. The code and data are released here: https://github.com/mayhugotong/GenTKG.
</details>
<details>
<summary><u><strong>Up To Date: Automatic Updating Knowledge Graphs Using LLMs (2024)</strong></u></summary>

**Authors:** Shahenda Hatem, Ghada Khoriba, Mohamed H. Gad-Elrab, Mohamed ElHelw  
**Venue & Year:** Procedia Computer Science, 2024  
**Link:** [https://www.sciencedirect.com/science/article/pii/S1877050924030072](https://www.sciencedirect.com/science/article/pii/S1877050924030072)  
**Abstract:**  
Maintaining up-to-date knowledge graphs (KGs) is essential for enhancing the accuracy and relevance of artificial intelligence (AI) applications, especially with sensitive domains. Yet, major KGs are either manually maintained (e.g., Wikidata) or infrequently rebuilt (e.g., DBpedia & YAGO). Thus, they contain many outdated facts. The rise of Large Language Models (LLMs) reasoning and Augmented Retrieval Generation approaches (RAG) gives KGs an interface to other trusted sources. This paper introduces a methodology utilizing Large Language Models (LLMs) to validate and update KG facts automatically. In particular, we utilize LLM reasoning capabilities to determine potentially outdated facts. After that, we use RAG techniques to generate an accurate fix for the fact. Experimental results on several LLMs and real-world datasets demonstrate the ability of our approach to propose accurate fixes. In addition, our experiments highlight the efficacy of few-shot prompts over zero-shot prompts.
</details>
<details>
<summary><u><strong>Pre-trained Language Model with Prompts for Temporal Knowledge Graph Completion (2023)</strong></u></summary>

**Authors:** Wenjie Xu, Ben Liu, Miao Peng, Xu Jia, Min Peng  
**Venue & Year:** Findings of the Association for Computational Linguistics: ACL, 2023  
**Link:** [https://aclanthology.org/2023.findings-acl.493/](https://aclanthology.org/2023.findings-acl.493/)  
**Abstract:**  
Temporal Knowledge graph completion (TKGC) is a crucial task that involves reasoning at known timestamps to complete the missing part of facts and has attracted more and more attention in recent years. Most existing methods focus on learning representations based on graph neural networks while inaccurately extracting information from timestamps and insufficiently utilizing the implied information in relations. To address these problems, we propose a novel TKGC model, namely Pre-trained Language Model with Prompts for TKGC (PPT). We convert a series of sampled quadruples into pre-trained language model inputs and convert intervals between timestamps into different prompts to make coherent sentences with implicit semantic information. We train our model with a masking strategy to convert TKGC task into a masked token prediction task, which can leverage the semantic information in pre-trained language models. Experiments on three benchmark datasets and extensive analysis demonstrate that our model has great competitiveness compared to other models with four metrics. Our model can effectively incorporate information from temporal knowledge graphs into the language models.
</details>
<details>
<summary><u><strong>Large Language Models-guided Dynamic Adaptation for Temporal Knowledge Graph Reasoning (2024)</strong></u></summary>

**Authors:** Jiapu Wang, Kai Sun, Linhao Luo, Wei Wei, Yongli Hu, Alan Wee-Chung Liew, Shirui Pan, Baocai Yin  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2405.14170](https://arxiv.org/abs/2405.14170)  
**Abstract:**  
Temporal Knowledge Graph Reasoning (TKGR) is the process of utilizing temporal information to capture complex relations within a Temporal Knowledge Graph (TKG) to infer new knowledge. Conventional methods in TKGR typically depend on deep learning algorithms or temporal logical rules. However, deep learning-based TKGRs often lack interpretability, whereas rule-based TKGRs struggle to effectively learn temporal rules that capture temporal patterns. Recently, Large Language Models (LLMs) have demonstrated extensive knowledge and remarkable proficiency in temporal reasoning. Consequently, the employment of LLMs for Temporal Knowledge Graph Reasoning (TKGR) has sparked increasing interest among researchers. Nonetheless, LLMs are known to function as black boxes, making it challenging to comprehend their reasoning process. Additionally, due to the resource-intensive nature of fine-tuning, promptly updating LLMs to integrate evolving knowledge within TKGs for reasoning is impractical. To address these challenges, in this paper, we propose a Large Language Models-guided Dynamic Adaptation (LLM-DA) method for reasoning on TKGs. Specifically, LLM-DA harnesses the capabilities of LLMs to analyze historical data and extract temporal logical rules. These rules unveil temporal patterns and facilitate interpretable reasoning. To account for the evolving nature of TKGs, a dynamic adaptation strategy is proposed to update the LLM-generated rules with the latest events. This ensures that the extracted rules always incorporate the most recent knowledge and better generalize to the predictions on future events. Experimental results show that without the need of fine-tuning, LLM-DA significantly improves the accuracy of reasoning over several common datasets, providing a robust framework for TKGR tasks.
</details>
<details>
<summary><u><strong>Back to the Future: Towards Explainable Temporal Reasoning with Large Language Models (2024)</strong></u></summary>

**Authors:** Chenhan Yuan, Qianqian Xie, Jimin Huang, Sophia Ananiadou  
**Venue & Year:** ACM, 2024  
**Link:** [https://doi.org/10.1145/3589334.3645376](https://doi.org/10.1145/3589334.3645376)  
**Abstract:**  
Temporal reasoning is a crucial natural language processing (NLP) task, providing a nuanced understanding of time-sensitive contexts within textual data. Although recent advancements in Large Language Models (LLMs) have demonstrated their potential in temporal reasoning, the predominant focus has been on tasks such as temporal expression detection, normalization, and temporal relation extraction. These tasks are primarily designed for the extraction of direct and past temporal cues from given contexts and to engage in simple reasoning processes. A significant gap remains when considering complex reasoning tasks such as event forecasting, which requires multi-step temporal reasoning on events and prediction on the future timestamp. Another notable limitation of existing methods is their incapability to illustrate their reasoning process for explaining their prediction, hindering explainability. In this paper, we introduce the first task of explainable temporal reasoning, to predict an event's occurrence at a future timestamp based on context which requires multiple reasoning over multiple events, and subsequently provide a clear explanation for their prediction. Our task offers a comprehensive evaluation of both the LLMs' complex temporal reasoning ability, the future event prediction ability, and explainability-a critical attribute for AI applications. To support this task, we present the first instruction-tuning dataset of explainable temporal reasoning (ExpTime) with 26k derived from the temporal knowledge graph datasets, using a novel knowledge-graph-instructed-generation strategy. Based on the dataset, we propose the first open-source LLM series TimeLlaMA based on the foundation LLM LlaMA2, with the ability of instruction following for explainable temporal reasoning. We compare the performance of our method and a variety of LLMs, where our method achieves the state-of-the-art performance of temporal prediction and explanation generation. We also explore the impact of instruction tuning and different training sizes of instruction-tuning data, highlighting LLM's capabilities and limitations in complex temporal prediction and explanation generation.
</details>
<details>
<summary><u><strong>RealTCD: Temporal Causal Discovery from Interventional Data with Large Language Model (2024)</strong></u></summary>

**Authors:** Peiwen Li, Xin Wang, Zeyang Zhang, Yuan Meng, Fang Shen, Yue Li, Jialong Wang, Yang Li, Wenwu Zhu  
**Venue & Year:** ACM, 2024  
**Link:** [https://doi.org/10.1145/3627673.3680042](https://doi.org/10.1145/3627673.3680042)  
**Abstract:**  
In the field of Artificial Intelligence for Information Technology Operations, causal discovery is pivotal for operation and maintenance of systems, facilitating downstream industrial tasks such as root cause analysis. Temporal causal discovery, as an emerging method, aims to identify temporal causal relations between variables directly from observations by utilizing interventional data. However, existing methods mainly focus on synthetic datasets with heavy reliance on interventional targets and ignore the textual information hidden in real-world systems, failing to conduct causal discovery for real industrial scenarios. To tackle this problem, in this paper we investigate temporal causal discovery in industrial scenarios, which faces two critical challenges: how to discover causal relations without the interventional targets that are costly to obtain in practice, and how to discover causal relations via leveraging the textual information in systems which can be complex yet abundant in industrial contexts. To address these challenges, we propose the RealTCD framework, which is able to leverage domain knowledge to discover temporal causal relations without interventional targets. We first develop a score-based temporal causal discovery method capable of discovering causal relations without relying on interventional targets through strategic masking and regularization. Then, by employing Large Language Models (LLMs) to handle texts and integrate domain knowledge, we introduce LLM-guided meta-initialization to extract the meta-knowledge from textual information hidden in systems to boost the quality of discovery. We conduct extensive experiments on both simulation datasets and our real-world application scenario to show the superiority of our proposed RealTCD over existing baselines in temporal causal discovery.
</details>
<details>
<summary><u><strong>DynLLM: When Large Language Models Meet Dynamic Graph Recommendation (2024)</strong></u></summary>

**Authors:** Ziwei Zhao, Fake Lin, Xi Zhu, Zhi Zheng, Tong Xu, Shitian Shen, Xueying Li, Zikai Yin, Enhong Chen  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2405.07580](https://arxiv.org/abs/2405.07580)  
**Abstract:**  
Last year has witnessed the considerable interest of Large Language Models (LLMs) for their potential applications in recommender systems, which may mitigate the persistent issue of data sparsity. Though large efforts have been made for user-item graph augmentation with better graph-based recommendation performance, they may fail to deal with the dynamic graph recommendation task, which involves both structural and temporal graph dynamics with inherent complexity in processing time-evolving data. To bridge this gap, in this paper, we propose a novel framework, called DynLLM, to deal with the dynamic graph recommendation task with LLMs. Specifically, DynLLM harnesses the power of LLMs to generate multi-faceted user profiles based on the rich textual features of historical purchase records, including crowd segments, personal interests, preferred categories, and favored brands, which in turn supplement and enrich the underlying relationships between users and items. Along this line, to fuse the multi-faceted profiles with temporal graph embedding, we engage LLMs to derive corresponding profile embeddings, and further employ a distilled attention mechanism to refine the LLM-generated profile embeddings for alleviating noisy signals, while also assessing and adjusting the relevance of each distilled facet embedding for seamless integration with temporal graph embedding from continuous time dynamic graphs (CTDGs). Extensive experiments on two real e-commerce datasets have validated the superior improvements of DynLLM over a wide range of state-of-the-art baseline methods.
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



<details>
<summary><u><strong>DARG: Dynamic Evaluation of Large Language Models via Adaptive Reasoning Graph (2024)</strong></u></summary>

**Authors:** Zhehao Zhang, Jiaao Chen, Diyi Yang  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2406.17271](https://arxiv.org/abs/2406.17271)  
**Abstract:**  
The current paradigm of evaluating Large Language Models (LLMs) through static benchmarks comes with significant limitations, such as vulnerability to data contamination and a lack of adaptability to the evolving capabilities of LLMs. Therefore, evaluation methods that can adapt and generate evaluation data with controlled complexity are urgently needed. In this work, we introduce Dynamic Evaluation of LLMs via Adaptive Reasoning Graph Evolvement (DARG) to dynamically extend current benchmarks with controlled complexity and diversity. Specifically, we first extract the reasoning graphs of data points in current benchmarks and then perturb the reasoning graphs to generate novel testing data. Such newly generated test samples can have different levels of complexity while maintaining linguistic diversity similar to the original benchmarks. We further use a code-augmented LLM to ensure the label correctness of newly generated data. We apply our DARG framework to diverse reasoning tasks in four domains with 15 state-of-the-art LLMs. Experimental results show that almost all LLMs experience a performance decrease with increased complexity and certain LLMs exhibit significant drops. Additionally, we find that LLMs exhibit more biases when being evaluated via the data generated by DARG with higher complexity levels. These observations provide useful insights into how to dynamically and adaptively evaluate LLMs. The code is available at this https URL.
</details>
<details>
<summary><u><strong>AnomalyLLM: Few-shot Anomaly Edge Detection for Dynamic Graphs using Large Language Models (2024)</strong></u></summary>

**Authors:** Shuo Liu, Di Yao, Lanting Fang, Zhetao Li, Wenbin Li, Kaiyu Feng, XiaoWen Ji, Jingping Bi  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2405.07626](https://arxiv.org/abs/2405.07626)  
**Abstract:**  
Detecting anomaly edges for dynamic graphs aims to identify edges significantly deviating from the normal pattern and can be applied in various domains, such as cybersecurity, financial transactions and AIOps. With the evolving of time, the types of anomaly edges are emerging and the labeled anomaly samples are few for each type. Current methods are either designed to detect randomly inserted edges or require sufficient labeled data for model training, which harms their applicability for real-world applications. In this paper, we study this problem by cooperating with the rich knowledge encoded in large language models(LLMs) and propose a method, namely AnomalyLLM. To align the dynamic graph with LLMs, AnomalyLLM pre-trains a dynamic-aware encoder to generate the representations of edges and reprograms the edges using the prototypes of word embeddings. Along with the encoder, we design an in-context learning framework that integrates the information of a few labeled samples to achieve few-shot anomaly detection. Experiments on four datasets reveal that AnomalyLLM can not only significantly improve the performance of few-shot anomaly detection, but also achieve superior results on new anomalies without any update of model parameters.
</details>
<details>
<summary><u><strong>Language-Grounded Dynamic Scene Graphs for Interactive Object Search With Mobile Manipulation (2024)</strong></u></summary>

**Authors:** Daniel Honerkamp, Martin Büchner, Fabien Despinoy, Tim Welschehold, Abhinav Valada  
**Venue & Year:** IEEE Robotics and Automation Letters, 2024  
**Link:** [https://doi.org/10.1109/LRA.2024.3441495](https://doi.org/10.1109/LRA.2024.3441495)  
**Abstract:**  
To fully leverage the capabilities of mobile manipulation robots, it is imperative that they are able to autonomously execute long-horizon tasks in large unexplored environments. While large language models (LLMs) have shown emergent reasoning skills on arbitrary tasks, existing work primarily concentrates on explored environments, typically focusing on either navigation or manipulation tasks in isolation. In this work, we propose MoMa-LLM, a novel approach that grounds language models within structured representations derived from open-vocabulary scene graphs, dynamically updated as the environment is explored. We tightly interleave these representations with an object-centric action space. Given object detections, the resulting approach is zero-shot, open-vocabulary, and readily extendable to a spectrum of mobile manipulation and household robotic tasks. We demonstrate the effectiveness of MoMa-LLM in a novel semantic interactive search task in large realistic indoor environments. In extensive experiments in both simulation and the real world, we show substantially improved search efficiency compared to conventional baselines and state-of-the-art approaches, as well as its applicability to more abstract tasks.
</details>
<details>
<summary><u><strong>Temporal Relational Reasoning of Large Language Models for Detecting Stock Portfolio Crashes (2024)</strong></u></summary>

**Authors:** Kelvin J. L. Koa, Yunshan Ma, Ritchie Ng, Huanhuan Zheng, Tat-Seng Chua  
**Venue & Year:** arXiv preprint, 2024  
**Link:** [https://arxiv.org/abs/2410.17266](https://arxiv.org/abs/2410.17266)  
**Abstract:**  
Stock portfolios are often exposed to rare consequential events (e.g., 2007 global financial crisis, 2020 COVID-19 stock market crash), as they do not have enough historical information to learn from. Large Language Models (LLMs) now present a possible tool to tackle this problem, as they can generalize across their large corpus of training data and perform zero-shot reasoning on new events, allowing them to detect possible portfolio crash events without requiring specific training data. However, detecting portfolio crashes is a complex problem that requires more than basic reasoning abilities. Investors need to dynamically process the impact of each new information found in the news articles, analyze the the relational network of impacts across news events and portfolio stocks, as well as understand the temporal context between impacts across time-steps, in order to obtain the overall aggregated effect on the target portfolio. In this work, we propose an algorithmic framework named Temporal Relational Reasoning (TRR). It seeks to emulate the spectrum of human cognitive capabilities used for complex problem-solving, which include brainstorming, memory, attention and reasoning. Through extensive experiments, we show that TRR is able to outperform state-of-the-art solutions on detecting stock portfolio crashes, and demonstrate how each of the proposed components help to contribute to its performance through an ablation study. Additionally, we further explore the possible applications of TRR by extending it to other related complex problems, such as the detection of possible global crisis events in Macroeconomics.
</details>
<details>
<summary><u><strong>Dynamic Benchmarking of Masked Language Models on Temporal Concept Drift with Multiple Views (2023)</strong></u></summary>

**Authors:** Katerina Margatina, Shuai Wang, Yogarshi Vyas, Neha Anna John, Yassine Benajiba, Miguel Ballesteros  
**Venue & Year:** Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, 2023  
**Link:** [https://aclanthology.org/2023.eacl-main.211/](https://aclanthology.org/2023.eacl-main.211/)  
**Abstract:**  
Temporal concept drift refers to the problem of data changing over time. In the field of NLP, that would entail that language (e.g. new expressions, meaning shifts) and factual knowledge (e.g. new concepts, updated facts) evolve over time. Focusing on the latter, we benchmark 11 pretrained masked language models (MLMs) on a series of tests designed to evaluate the effect of temporal concept drift, as it is crucial that widely used language models remain up-to-date with the ever-evolving factual updates of the real world. Specifically, we provide a holistic framework that (1) dynamically creates temporal test sets of any time granularity (e.g. month, quarter, year) of factual data from Wikidata, (2) constructs fine-grained splits of tests (e.g. updated, new, unchanged facts) to ensure comprehensive analysis, and (3) evaluates MLMs in three distinct ways (single-token probing, multi-token generation, MLM scoring). In contrast to prior work, our framework aims to unveil how robust an MLM is over time and thus to provide a signal in case it has become outdated, by leveraging multiple views of evaluation.
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
  author={Mengran Li, Pengyu Zhang, Wenbin Xing, Yijia Zheng, Klim Zaporojets, Junzhou Chen, Ronghui Zhang, Yong Zhang, Siyuan Gong, Jia Hu, Xiaolei Ma, Zhiyuan Liu, Paul Groth, Marcel Worring},
  journal={Expert Systems with Applications},
  pages={129643},
  year={2025},
  publisher={Elsevier}
}

@article{li2025using,
  title={Using Large Language Models to Tackle Fundamental Challenges in Graph Learning: A Comprehensive Survey},
  author={Mengran Li, Pengyu Zhang, Wenbin Xing, Yijia Zheng, Klim Zaporojets, Junzhou Chen, Ronghui Zhang, Yong Zhang, Siyuan Gong, Jia Hu, Xiaolei Ma, Zhiyuan Liu, Paul Groth, Marcel Worring},
  journal={arXiv preprint arXiv:2505.18475},
  year={2025}
}
