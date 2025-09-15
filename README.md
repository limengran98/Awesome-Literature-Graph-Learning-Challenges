
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

| Title | Authors | Venue & Year | Link |
|-------|---------|---------------|------|
| Spatiotemporal Pre-Trained Large Language Model for Forecasting With Missing Values | Le Fang, Wei Xiang, Shirui Pan, Flora D. Salim, Yi-Ping Phoebe Chen | IEEE Internet of Things Journal, 2025 | [Link](https://doi.org/10.1109/JIOT.2024.3524030) |
| LLM as Prompter: Low-resource Inductive Reasoning on Arbitrary Knowledge Graphs | Kai Wang, Yuwei Xu, Zhiyong Wu, Siqiang Luo | Findings of the Association for Computational Linguistics, 2024 | [Link](https://doi.org/10.18653/v1/2024.findings-acl.224) |
| On LLM-Enhanced Mixed-Type Data Imputation with High-Order Message Passing | Jianwei Wang, Kai Wang, Ying Zhang, Wenjie Zhang, Xiwei Xu, Xuemin Lin | arXiv preprint, 2025 | [Link](https://arxiv.org/abs/2501.02191) |
| Large language models as topological structure enhancers for text-attributed graphs | Shengyin Sun, Yuxiang Ren, Chen Ma, Xuecang Zhang | arXiv preprint, 2023 | [Link](https://doi.org/10.48550/arXiv.2311.14324) |
| Empower text-attributed graphs learning with large language models (LLMs) | Jianxiang Yu, Yuxiang Ren, Chenghua Gong, Jiaqi Tan, Xiang Li, Xuecang Zhang | arXiv preprint, 2023 | [Link](https://doi.org/10.48550/arXiv.2310.09872) |
| Label-free node classification on graphs with large language models (LLMs) | Zhikai Chen, Haitao Mao, Hongzhi Wen, Haoyu Han, Wei Jin, Haiyang Zhang, Hui Liu, Jiliang Tang | arXiv preprint, 2023 | [Link](https://arxiv.org/abs/2310.04668) |
| GraphLLM: Boosting graph reasoning ability of large language models | Ziwei Chai, Tianjie Zhang, Liang Wu, Kaiqiao Han, Xiaohai Hu, Xuanwen Huang, Yang Yang | arXiv preprint, 2023 | [Link](https://doi.org/10.48550/arXiv.2310.05845) |


---

###  Few-shot Graph Learning

| Title | Authors | Venue & Year | Link |
|-------|---------|---------------|------|
| LLM-Empowered Few-Shot Node Classification on Incomplete Graphs with Real Node Degrees | Yun Li, Yi Yang, Jiaqi Zhu, Hui Chen, Hongan Wang | Proceedings of the ACM International Conference on Information and Knowledge Management, 2024 | [Link](https://doi.org/10.1145/3627673.3679861) |
| LinkGPT: Teaching Large Language Models to Predict Missing Links | Zhongmou He, Jing Zhu, Shengyi Qian, Joyce Chai, Danai Koutra | arXiv preprint, 2024 | [Link](https://doi.org/10.48550/arXiv.2406.04640) |
| AnomalyLLM: Few-shot Anomaly Edge Detection for Dynamic Graphs using Large Language Models | Shuo Liu, Di Yao, Lanting Fang, Zhetao Li, Wenbin Li, Kaiyu Feng, XiaoWen Ji, Jingping Bi | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2405.07626) |
| HeGTa: Leveraging Heterogeneous Graph-enhanced Large Language Models for Few-shot Complex Table Understanding | Rihui Jin, Yu Li, Guilin Qi, Nan Hu, Yuan-Fang Li, Jiaoyan Chen, Jianan Wang, Yongrui Chen, Dehai Min, Sheng Bi | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2403.19723) |
| FlexKBQA: A Flexible LLM-powered Framework for Few-shot Knowledge Base Question Answering | Zhenyu Li, Sunqi Fan, Yu Gu, Xiuxing Li, Zhichao Duan, Bowen Dong, Ning Liu, Jianyong Wang | Proceedings of the AAAI Conference on Artificial Intelligence, 2024 | [Link](https://doi.org/10.1609/aaai.v38i17.29823) |
| Zero-shot Knowledge Graph Question Generation via Multi-agent LLMs and Small Models Synthesis | Runhao Zhao, Jiuyang Tang, Weixin Zeng, Ziyang Chen, Xiang Zhao | Proceedings of the ACM International Conference on Information and Knowledge Management, 2024 | [Link](https://doi.org/10.1145/3627673.3679805) |

---

###  Knowledge Graph Completion

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

---

### Datasets, Metrics, and Tasks (Incompleteness)
<div style="overflow-x:auto;"> <table> <thead> <tr> <th>Domain</th> <th>Incompleteness</th> <th>Method</th> <th>Typical Datasets</th> <th>Common Metrics</th> <th>Downstream Tasks</th> </tr> </thead> <tbody> <tr><td>Robust Graph Learning</td><td>Node</td><td><b>LLM4NG</b></td><td>Cora, PubMed, ogbn-arxiv</td><td>Accuracy</td><td>Node Classification</td></tr> <tr><td>Robust Graph Learning</td><td>Node</td><td><b>LLM-TAG</b></td><td>Cora, Citeseer, PubMed, Arxiv-2023</td><td>Accuracy</td><td>Node Classification</td></tr> <tr><td>Robust Graph Learning</td><td>Node</td><td><b>SPLLM</b></td><td>PeMS03, PeMS04, PeMS07</td><td>MAE, RMSE, MAPE</td><td>Spatiotemporal Forecasting</td></tr> <tr><td>Robust Graph Learning</td><td>Label</td><td><b>LLMGNN</b></td><td>Cora, Citeseer, PubMed, ogbn-arxiv, ogbn-products, WikiCS</td><td>Accuracy</td><td>Node Classification</td></tr> <tr><td>Robust Graph Learning</td><td>Mixed</td><td><b>GraphLLM</b></td><td>Synthetic Data</td><td>Exact Match Accuracy</td><td>Graph Reasoning</td></tr> <tr><td>Robust Graph Learning</td><td>Mixed</td><td><b>PROLINK</b></td><td>FB15k-237, Wikidata68K, NELL-995</td><td>MRR, Hits@N</td><td>Knowledge Graph Completion</td></tr> <tr><td>Robust Graph Learning</td><td>Mixed</td><td><b>UnIMP</b></td><td>BG, ZO, PK, BK, CS, ST, PW, BY, RR, WM</td><td>RMSE, MAE</td><td>Data Imputation</td></tr>
<tr><td>Few-Shot Graph Learning</td><td>Structure</td><td><b>LinkGPT</b></td><td>AmazonSports, Amazon-Clothing, MAG-Geology, MAG-Math</td><td>MRR, Hits@N</td><td>Link Prediction</td></tr>
<tr><td>Few-Shot Graph Learning</td><td>Structure</td><td><b>AnomalyLLM</b></td><td>UCI Messages, Blogcatalog, T-Finance, T-Social</td><td>AUC</td><td>Anomaly Detection</td></tr>
<tr><td>Few-Shot Graph Learning</td><td>Mixed</td><td><b>LLMDGCN</b></td><td>Cora, Citeseer, PubMed, Religion</td><td>Accuracy</td><td>Node Classification</td></tr>
<tr><td>Few-Shot Graph Learning</td><td>Mixed</td><td><b>HeGTa</b></td><td>IM-TQA, WCC, HiTab, WTQ, TabFact</td><td>Macro-F1, Accuracy</td><td>Table Understanding</td></tr>
<tr><td>Few-Shot Graph Learning</td><td>Mixed</td><td><b>FlexKBQA</b></td><td>GrailQA, WebQSP, KQA Pro</td><td>Exact Match, F1, Accuracy</td><td>Knowledge Graph Question Answering</td></tr>
<tr><td>Few-Shot Graph Learning</td><td>Mixed</td><td><b>KGQG</b></td><td>WebQuestions, PathQuestions</td><td>BLEU-4, ROUGE-L, Hits@N</td><td>Knowledge Graph Question Answering</td></tr>

<tr><td>Knowledge Graph Completion</td><td>Node</td><td><b>LLM-KGC</b></td><td>ILPC</td><td>MRR, Hits@N</td><td>Knowledge Graph Completion</td></tr>
<tr><td>Knowledge Graph Completion</td><td>Node</td><td><b>GS-KGC</b></td><td>WN18RR, FB15k-237, FB15k-237N, ICEWS14, ICEWS05-15</td><td>Hits@N</td><td>Knowledge Graph Completion</td></tr>
<tr><td>Knowledge Graph Completion</td><td>Node</td><td><b>GLTW</b></td><td>FB15k-237, WN18RR, Wikidata5M</td><td>MRR, Hits@N</td><td>Link Prediction</td></tr>
<tr><td>Knowledge Graph Completion</td><td>Label</td><td><b>KGs-LLM</b></td><td>Wikipedia</td><td>F1, Precision, Recall</td><td>Knowledge Graph Generation</td></tr>
<tr><td>Knowledge Graph Completion</td><td>Mixed</td><td><b>FSKG</b></td><td>WN18RR, FB15k-237</td><td>MRR, Hits@N</td><td>Knowledge Graph Completion</td></tr>
<tr><td>Knowledge Graph Completion</td><td>Mixed</td><td><b>KGLLM</b></td><td>WN11, FB13, WN18RR, YAGO3-10</td><td>Accuracy, MRR, Hits@N</td><td>Link Prediction, Knowledge Graph Completion</td></tr>
<tr><td>Knowledge Graph Completion</td><td>Mixed</td><td><b>KICGPT</b></td><td>FB15k-237, WN18RR</td><td>MRR, Hits@N</td><td>Link Prediction</td></tr>
<tr><td>Knowledge Graph Completion</td><td>Mixed</td><td><b>RL-LLM</b></td><td>Electronics, Instacart</td><td>Precision, Recall, Accuracy</td><td>Knowledge Graph Completion</td></tr>
<tr><td>Knowledge Graph Completion</td><td>Mixed</td><td><b>GoG</b></td><td>Synthetic Data</td><td>Hits@N</td><td>Knowledge Graph Question Answering</td></tr>
<tr><td>Knowledge Graph Completion</td><td>Mixed</td><td><b>KoPA</b></td><td>UMLS, CoDeX-S, FB15K-237N</td><td>F1, Precision, Recall, Accuracy</td><td>Knowledge Graph Completion</td></tr>
<tr><td>Knowledge Graph Completion</td><td>Mixed</td><td><b>LLMKG</b></td><td>Templates Easy, Templates Hard</td><td>Strict Metrics, Flexible Metrics</td><td>Knowledge Graph Completion</td></tr>
<tr><td>Knowledge Graph Completion</td><td>Mixed</td><td><b>DIFT</b></td><td>WN18RR, FB15k-237</td><td>MRR, Hits@N</td><td>Link Prediction, Knowledge Graph Completion</td></tr>
<tr><td>Knowledge Graph Completion</td><td>Mixed</td><td><b>CP-KGC</b></td><td>WN18RR, FB15k-237, UMLS</td><td>MRR, Hits@N</td><td>Knowledge Graph Completion</td></tr>
<tr><td>Knowledge Graph Completion</td><td>Mixed</td><td><b>MuKDC</b></td><td>NELL, Wiki</td><td>MRR, Hits@N</td><td>Knowledge Graph Completion</td></tr>

</tbody> </table> </div>


---

## Imbalance in Graphs

> Real-world graphs often exhibit skewed class distributions or unbalanced structural patterns, making training difficult and biased.

### Class-Imbalanced Graph Learning


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


---


###  Structure-Imbalanced Graph Learning

| Title | Authors | Venue & Year | Link |
|-------|---------|---------------|------|
| Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks? | Zhongjian Zhang, Xiao Wang, Huichi Zhou, Yue Yu, Mengmei Zhang, Cheng Yang, Chuan Shi | arXiv preprint, 2024 | [Link](https://doi.org/10.48550/arXiv.2408.08685) |
| Subgraph-Aware Training of Language Models for Knowledge Graph Completion Using Structure-Aware Contrastive Learning | Youmin Ko, Hyemin Yang, Taeuk Kim, Hyunjoon Kim | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2407.12703) |
| Multi-perspective improvement of knowledge graph completion with large language models | Derong Xu, Ziheng Zhang, Zhenxi Lin, Xian Wu, Zhihong Zhu, Tong Xu, Xiangyu Zhao, Yefeng Zheng, Enhong Chen | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2403.01972) |
| Graphedit: Large language models for graph structure learning | Zirui Guo, Lianghao Xia, Yanhua Yu, Yuling Wang, Zixuan Yang, Wei Wei, Liang Pang, Tat-Seng Chua, Chao Huang | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2402.15183) |

---

### Datasets, Metrics, and Tasks (Imbalance)

<div style="overflow-x:auto;"> <table> <thead> <tr> <th>Domain</th> <th>Tasks</th> <th>Method</th> <th>Typical Datasets</th> <th>Common Metrics</th> <th>Downstream Tasks</th> </tr> </thead> <tbody> <tr><td>Class Imbalance Graph Learning</td><td>Node Classification</td><td><b>LLM4NG</b></td><td>Cora, PubMed, ogbn-arxiv</td><td>Accuracy</td><td>Few-shot Node Classification</td></tr> <tr><td>Class Imbalance Graph Learning</td><td>Node Classification</td><td><b>LLM-GNN</b></td><td>Cora, Citeseer, PubMed, ogbn-arxiv, ogbn-products, WikiCS</td><td>Accuracy</td><td>Label-free Node Classification</td></tr> <tr><td>Class Imbalance Graph Learning</td><td>Node Classification</td><td><b>G2P2</b></td><td>Cora; Amazon (Art, Industrial, Music Instruments)</td><td>Accuracy, Macro-F1</td><td>Zero-/Few-shot Low-resource Text Classification</td></tr> <tr><td>Class Imbalance Graph Learning</td><td>Node Classification</td><td><b>LA-TAG</b></td><td>Cora, PubMed, Photo, Computer, Children</td><td>Accuracy, Macro-F1</td><td>Zero-/Few-shot Low-resource Text Classification</td></tr> <tr><td>Class Imbalance Graph Learning</td><td>Node Classification</td><td><b>GSS-Net</b></td><td>Amazon (Magazine Subscriptions, Appliances, Gift Cards)</td><td>Accuracy, Precision, Recall, F1, MSE, RMSE, MAE</td><td>Sentiment on Streaming E-commerce Reviews</td></tr> <tr><td>Class Imbalance Graph Learning</td><td>Node Classification</td><td><b>TAGrader</b></td><td>Cora, PubMed, ogbn-products, Arxiv-2023</td><td>Accuracy</td><td>Node Classification on TAGs</td></tr> <tr><td>Class Imbalance Graph Learning</td><td>Node Classification</td><td><b>SEGA</b></td><td>DAIC-WOZ, EATD</td><td>Macro-F1</td><td>Depression Detection</td></tr> <tr><td>Class Imbalance Graph Learning</td><td>Node Classification</td><td><b>SocioHyperNet</b></td><td>MBTI</td><td>Accuracy, AUC, Macro-F1, Micro-F1, IMP</td><td>Personality Traits</td></tr> <tr><td>Class Imbalance Graph Learning</td><td>Node Classification</td><td><b>Cella</b></td><td>Cora, Citeseer, PubMed, Wiki-CS</td><td>Accuracy, NMI, ARI, F1</td><td>Label-free Node Classification</td></tr> <tr><td>Class Imbalance Graph Learning</td><td>Node Classification</td><td><b>LLM-TIKG</b></td><td>threat-dataset</td><td>Precision, Recall, F1</td><td>Threat Intelligence KG Construction</td></tr> <tr><td>Class Imbalance Graph Learning</td><td>Node Classification</td><td><b>ANLM-assInNNER</b></td><td>NE dataset</td><td>Precision, Recall, F1</td><td>Robotic Fault Diagnosis KG Construction</td></tr> <tr><td>Class Imbalance Graph Learning</td><td>Node Classification</td><td><b>LLM-HetGDT</b></td><td>Twitter-HetDrug</td><td>Macro-F1, GMean</td><td>Online Drug Trafficking Detection</td></tr>
<tr><td>Class Imbalance Graph Learning</td><td>Prediction</td><td><b>LLM-SBCL</b></td><td>biology, law, cardiff20102, sydney19351, sydney23146</td><td>Binary-F1, Micro-F1, Macro-F1, Accuracy</td><td>Student Performance Prediction</td></tr>
<tr><td>Class Imbalance Graph Learning</td><td>Prediction</td><td><b>LKPNR</b></td><td>MIND</td><td>AUC, MRR, nDCG</td><td>Personalized News Recommendation</td></tr>
<tr><td>Class Imbalance Graph Learning</td><td>Prediction</td><td><b>LLM-DDA</b></td><td>BCFR-dataset</td><td>AUC, AUPR, F1, Precision</td><td>Computational Drug Repositioning</td></tr>

<tr><td>Class Imbalance Graph Learning</td><td>Graph Completion</td><td><b>KICGPT</b></td><td>FB15k-237, WN18RR</td><td>MRR, Hits@N</td><td>Link Completion</td></tr>
<tr><td>Class Imbalance Graph Learning</td><td>Graph Completion</td><td><b>KGCD</b></td><td>WN18RR, YAGO3-10, WN18</td><td>MRR, Hits@N</td><td>Low-resource Knowledge Graph Completion</td></tr>

<tr><td>Class Imbalance Graph Learning</td><td>Foundation Model</td><td><b>GraphCLIP</b></td><td>ogbn-arXiv, Arxiv-2023, PubMed, ogbn-products, Reddit, Cora, CiteSeer, Ele-Photo, Ele-Computers, Books-History, WikiCS, Instagram</td><td>Accuracy</td><td>Transfer Learning on TAGs</td></tr>

<tr><td>Structure Imbalance Graph Learning</td><td>Node Classification</td><td><b>GraphEdit</b></td><td>Cora, Citeseer, PubMed</td><td>Accuracy</td><td>Refining Graph Topologies</td></tr>
<tr><td>Structure Imbalance Graph Learning</td><td>Graph Completion</td><td><b>SATKGC</b></td><td>WN18RR, FB15k-237, Wikidata5M</td><td>MRR, Hits@N</td><td>Knowledge Graph Completion</td></tr>
<tr><td>Structure Imbalance Graph Learning</td><td>Graph Completion</td><td><b>MPIKGC</b></td><td>FB15k-237, WN18RR, FB13, WN11</td><td>MR, MRR, Hits@N, Accuracy</td><td>Knowledge Graph Completion</td></tr>
<tr><td>Structure Imbalance Graph Learning</td><td>Graph Completion</td><td><b>LLM4RGNN</b></td><td>Cora, Citeseer, PubMed, ogbn-arxiv, ogbn-products</td><td>Accuracy</td><td>Improving the Adversarial Robustness</td></tr>

</tbody> </table> </div>

---

## Cross-Domain Heterogeneity in Graphs

> Graphs with heterogeneous node/edge types, multimodal attributes, or domain-specific patterns require specialized methods.

###  Text-Attributed Graph Learning

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

---

###  Multimodal Attributed Graph Learning

| Title | Authors | Venue & Year | Link |
|-------|---------|---------------|------|
| UniGraph2: Learning a Unified Embedding Space to Bind Multimodal Graphs | Yufei He, Yuan Sui, Xiaoxin He, Yue Liu, Yifei Sun, Bryan Hooi | arXiv preprint, 2025 | [Link](https://arxiv.org/abs/2502.00806) |
| LLMRec: Large language models with graph augmentation for recommendation | Wei Wei, Xubin Ren, Jiabin Tang, Qinyong Wang, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, Chao Huang | Proceedings of the 17th ACM International Conference on Web Search and Data Mining, 2024 | [Link](https://doi.org/10.1145/3616855.3635853) |
| Touchup-G: Improving feature representation through graph-centric finetuning | Jing Zhu, Xiang Song, Vassilis Ioannidis, Danai Koutra, Christos Faloutsos | Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2024 | [Link](https://doi.org/10.1145/3626772.3657978) |
| GraphAdapter: Tuning vision-language models with dual knowledge graph | Xin Li, Dongze Lian, Zhihe Lu, Jiawang Bai, Zhibo Chen, Xinchao Wang | Advances in Neural Information Processing Systems, 2024 | [Link](https://openreview.net/forum?id=YmEDnMynuO&noteId=0rFYtJNqHc) |
| When Graph meets Multimodal: Benchmarking and Meditating on Multimodal Attributed Graphs Learning | Hao Yan, Chaozhuo Li, Jun Yin, Zhigang Yu, Weihao Han, Mingzheng Li, Zhengxin Zeng, Hao Sun, Senzhang Wang | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2410.09132) |
| Multimodal Graph Learning for Generative Tasks | Minji Yoon, Jing Yu Koh, Bryan Hooi, Russ Salakhutdinov | NeurIPS 2023 Workshop: New Frontiers in Graph Learning, 2023 | [Link](https://openreview.net/forum?id=YILik4gFBk) |

---


###  Structural Heterogeneous Graph Learning


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

---
### Datasets, Metrics, and Tasks (Cross-domain Heterogeneity)

<div style="overflow-x:auto;"> <table> <thead> <tr> <th>Domains</th> <th>Tasks</th> <th>Methods</th> <th>Typical Datasets</th> <th>Common Metrics</th> <th>Downstream Tasks</th> </tr> </thead> <tbody> <tr><td>Text-Attributed Graph Learning</td><td>Textual Attribute Alignment</td><td><b>TAPE</b></td><td>Cora, PubMed, Arxiv-2023, ogbn-arxiv, ogbn-products</td><td>Accuracy</td><td>Node Classification</td></tr> <tr><td>Text-Attributed Graph Learning</td><td>Textual Attribute Alignment</td><td><b>LLMRec</b></td><td>MovieLens, Netflix</td><td>Recall, NDCG, Precision</td><td>Item Recommendation</td></tr> <tr><td>Text-Attributed Graph Learning</td><td>Textual Attribute Alignment</td><td><b>MINGLE</b></td><td>MIMIC-III, CRADLE</td><td>Accuracy, AUC, AUPR, F1</td><td>Node Classification</td></tr> <tr><td>Text-Attributed Graph Learning</td><td>Textual Attribute Alignment</td><td><b>GHGRL</b></td><td>IMDB, DBLP, ACM, Wiki-CS, IMDB-RIR, DBLP-RID</td><td>Macro-F1, Micro-F1</td><td>Node Classification</td></tr>
<tr><td>Text-Attributed Graph Learning</td><td>Graph Foundation Model</td><td><b>OFA</b></td><td>Cora, PubMed, ogbn-arxiv, Wiki-CS, MOLHIV, MOLPCBA, FB15K237, WN18RR, ChEMBL</td><td>Accuracy, AUC, AUPR</td><td>Node Classification, Link Prediction, Graph Classification</td></tr>
<tr><td>Text-Attributed Graph Learning</td><td>Graph Foundation Model</td><td><b>UniGraph</b></td><td>Cora, PubMed, ogbn-arxiv, ogbn-products, Wiki-CS, FB15K237, WN18RR, MOLHIV, MOLPCBA</td><td>AUC</td><td>Node Classification, Link Prediction, Graph Classification</td></tr>
<tr><td>Text-Attributed Graph Learning</td><td>Graph Foundation Model</td><td><b>BooG</b></td><td>Cora, PubMed, ogbn-arxiv, Wiki-CS, MOLHIV, MOLPCBA</td><td>AUC</td><td>Node Classification, Graph Classification</td></tr>
<tr><td>Text-Attributed Graph Learning</td><td>Graph Foundation Model</td><td><b>Hyper-FM</b></td><td>Cora-CA-Text, Cora-CC-Text, Pubmed-CA-Text, Pubmed-CC-Text, AminerText, Arxiv-Text, Movielens-Text, IMDB-Text, GoodBook-Text, PPI-Text</td><td>Accuracy</td><td>Node Classification</td></tr>

<tr><td>Multimodal Attributed Graph Learning</td><td>MLLM-based Multimodal Alignment</td><td><b>LLMRec</b></td><td>MovieLens, Netflix</td><td>Recall, NDCG, Precision</td><td>Item Recommendation</td></tr>
<tr><td>Multimodal Attributed Graph Learning</td><td>MLLM-based Multimodal Alignment</td><td><b>MAGB</b></td><td>Cora, Wiki-CS, Ele-Photo, Flickr, Movies, Toys, Grocery, Reddit-S, Reddit-M</td><td>Accuracy, F1</td><td>Node Classification</td></tr>

<tr><td>Multimodal Attributed Graph Learning</td><td>Graph-Enhanced Multimodal Alignment</td><td><b>MMGL</b></td><td>WikiWeb2M</td><td>BLEU-4, ROUGE-L, CIDEr</td><td>Section Summarization</td></tr>
<tr><td>Multimodal Attributed Graph Learning</td><td>Graph-Enhanced Multimodal Alignment</td><td><b>GraphAdapter</b></td><td>ImageNet, StanfordCars, UCF101, Caltech101, Flowers102, SUN397, DTD, EuroSAT, FGVCAircraft, OxfordPets, Food101</td><td>Accuracy</td><td>Image Classification</td></tr>
<tr><td>Multimodal Attributed Graph Learning</td><td>Graph-Enhanced Multimodal Alignment</td><td><b>TouchUp-G</b></td><td>ogbn-arxiv, ogbn-products, Books, Amazon-CP</td><td>MRR, Hits@N, Accuracy</td><td>Link Prediction, Node Classification</td></tr>
<tr><td>Multimodal Attributed Graph Learning</td><td>Graph-Enhanced Multimodal Alignment</td><td><b>UniGraph2</b></td><td>Cora, PubMed, ogbn-arxiv, ogbn-papers100M, ogbn-products, Wiki-CS, FB15K237, WN18RR, Amazon-Sports, Amazon-Cloth, Goodreads-LP, Goodreads-NC, Ele-Fashion, WikiWeb2M</td><td>Accuracy, BLEU-4, ROUGE-L, CIDEr</td><td>Node Classification, Edge Classification, Section Summarization</td></tr>

<tr><td>Structural Heterogeneous Graph Learning</td><td>Topological Graph Textualization</td><td><b>LLMtoGraph</b></td><td>Synthetic graph data</td><td>Accuracy, Positive Response Ratio</td><td>Node Classification, Path Finding, Pattern Matching</td></tr>
<tr><td>Structural Heterogeneous Graph Learning</td><td>Topological Graph Textualization</td><td><b>NLGraph</b></td><td>NLGraph</td><td>Accuracy, Partial Credit Score, Relative Error</td><td>Path Finding, Pattern Matching, Topological Sort</td></tr>
<tr><td>Structural Heterogeneous Graph Learning</td><td>Topological Graph Textualization</td><td><b>Talk like a Graph</b></td><td>GraphQA</td><td>Accuracy</td><td>Link Prediction, Pattern Matching</td></tr>
<tr><td>Structural Heterogeneous Graph Learning</td><td>Topological Graph Textualization</td><td><b>GPT4Graph</b></td><td>ogbn-arxiv, MOLHIV, MOLPCBA, MetaQA</td><td>Accuracy</td><td>Node Classification, Graph Classification, Graph Query Language Generation</td></tr>
<tr><td>Structural Heterogeneous Graph Learning</td><td>Topological Graph Textualization</td><td><b>GITA</b></td><td>GVLQA</td><td>Accuracy</td><td>Link Prediction, Pattern Matching, Path Finding, Topological Sort</td></tr>
<tr><td>Structural Heterogeneous Graph Learning</td><td>Topological Graph Textualization</td><td><b>LLM4-Hypergraph</b></td><td>LLM4Hypergraph</td><td>Accuracy</td><td>Isomorphism Recognition, Structure Classification, Link Prediction, Path Finding</td></tr>

<tr><td>Structural Heterogeneous Graph Learning</td><td>Attributed Graph Textualization</td><td><b>GraphText</b></td><td>Cora, Citeseer, Texas, Wisconsin, Cornell</td><td>Accuracy</td><td>Node Classification</td></tr>
<tr><td>Structural Heterogeneous Graph Learning</td><td>Attributed Graph Textualization</td><td><b>WalkLM</b></td><td>PubMed, MIMIC-III</td><td>Macro-F1, Micro-F1, AUC, MRR</td><td>Node Classification, Link Prediction</td></tr>
<tr><td>Structural Heterogeneous Graph Learning</td><td>Attributed Graph Textualization</td><td><b>Path-LLM</b></td><td>Cora, Citeseer, PubMed, ogbn-arxiv</td><td>Macro-F1, Micro-F1, AUC, Accuracy</td><td>Node Classification, Link Prediction</td></tr>
<tr><td>Structural Heterogeneous Graph Learning</td><td>Attributed Graph Textualization</td><td><b>InstructGLM</b></td><td>Cora, PubMed, ogbn-arxiv</td><td>Accuracy</td><td>Node Classification, Link Prediction</td></tr>
<tr><td>Structural Heterogeneous Graph Learning</td><td>Attributed Graph Textualization</td><td><b>MuseGraph</b></td><td>Cora, ogbn-arxiv, MIMIC-III, AGENDA, WebNLG</td><td>Macro-F1, Micro-F1, Weighted-F1, BLEU-4, METEOR, ROUGE-L, CHRF++</td><td>Node Classification, Graph-to-Text Generation</td></tr>
<tr><td>Structural Heterogeneous Graph Learning</td><td>Attributed Graph Textualization</td><td><b>Graph-LLM</b></td><td>Cora, Citeseer, PubMed, ogbn-arxiv, ogbn-products</td><td>Accuracy</td><td>Node Classification</td></tr>

<tr><td>Structural Heterogeneous Graph Learning</td><td>Graph Token Learning</td><td><b>GNP</b></td><td>OBQA, ARC, PIQA, Riddle, PQA, BioASQ</td><td>Accuracy</td><td>Question Answering</td></tr>
<tr><td>Structural Heterogeneous Graph Learning</td><td>Graph Token Learning</td><td><b>GraphToken</b></td><td>GraphQA</td><td>Accuracy</td><td>Link Prediction, Pattern Matching</td></tr>
<tr><td>Structural Heterogeneous Graph Learning</td><td>Graph Token Learning</td><td><b>GraphGPT</b></td><td>Cora, PubMed, ogbn-arxiv</td><td>Accuracy, Macro-F1, AUC</td><td>Node Classification, Link Prediction</td></tr>
<tr><td>Structural Heterogeneous Graph Learning</td><td>Graph Token Learning</td><td><b>LLaGA</b></td><td>Cora, PubMed, ogbn-arxiv, ogbn-products</td><td>Accuracy</td><td>Node Classification, Link Prediction</td></tr>
<tr><td>Structural Heterogeneous Graph Learning</td><td>Graph Token Learning</td><td><b>TEA-GLM</b></td><td>Cora, PubMed, ogbn-arxiv, TAG benchmark</td><td>Accuracy, AUC</td><td>Node Classification, Link Prediction</td></tr>
<tr><td>Structural Heterogeneous Graph Learning</td><td>Graph Token Learning</td><td><b>HiGPT</b></td><td>IMDB, DBLP, ACM</td><td>Macro-F1, Micro-F1, AUC</td><td>Node Classification</td></tr>

</tbody> </table> </div>
---

## Dynamic Instability in Graphs

> Graph structures may evolve over time or require adaptive interaction. These works explore LLMs in dynamic graph settings.

###  Querying and Reasoning

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

---

###  Generating and Updating

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

---

###  Evaluation and Application

| Title | Authors | Venue & Year | Link |
|-------|---------|---------------|------|
| DARG: Dynamic Evaluation of Large Language Models via Adaptive Reasoning Graph | Zhehao Zhang, Jiaao Chen, Diyi Yang | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2406.17271) |
| AnomalyLLM: Few-shot Anomaly Edge Detection for Dynamic Graphs using Large Language Models | Shuo Liu, Di Yao, Lanting Fang, Zhetao Li, Wenbin Li, Kaiyu Feng, XiaoWen Ji, Jingping Bi | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2405.07626) |
| Language-Grounded Dynamic Scene Graphs for Interactive Object Search With Mobile Manipulation | Daniel Honerkamp, Martin Büchner, Fabien Despinoy, Tim Welschehold, Abhinav Valada | IEEE Robotics and Automation Letters, 2024 | [Link](https://doi.org/10.1109/LRA.2024.3441495) |
| Temporal Relational Reasoning of Large Language Models for Detecting Stock Portfolio Crashes | Kelvin J. L. Koa, Yunshan Ma, Ritchie Ng, Huanhuan Zheng, Tat-Seng Chua | arXiv preprint, 2024 | [Link](https://arxiv.org/abs/2410.17266) |
| Dynamic Benchmarking of Masked Language Models on Temporal Concept Drift with Multiple Views | Katerina Margatina, Shuai Wang, Yogarshi Vyas, Neha Anna John, Yassine Benajiba, Miguel Ballesteros | Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, 2023 | [Link](https://aclanthology.org/2023.eacl-main.211/) |

---

### Datasets, Metrics, and Tasks (Dynamic Instability)

<div style="overflow-x:auto;"> <table> <thead> <tr> <th>Domain</th> <th>Category</th> <th>Method</th> <th>Typical Datasets</th> <th>Common Metrics</th> <th>Downstream Tasks</th> </tr> </thead> <tbody> <tr><td>Querying &amp; Reasoning</td><td>Forecasting &amp; Reasoning</td><td><b>ICL</b></td><td>WIKI, YAGO, ICEWS14, ICEWS18</td><td>MRR, Hits@N</td><td>Link Prediction</td></tr> <tr><td>Querying &amp; Reasoning</td><td>Forecasting &amp; Reasoning</td><td><b>zrLLM</b></td><td>ICEWS, ACLED</td><td>MRR, Hits@N</td><td>Link Prediction</td></tr> <tr><td>Querying &amp; Reasoning</td><td>Forecasting &amp; Reasoning</td><td><b>CoH</b></td><td>ICEWS14, ICEWS18, ICEWS05-15</td><td>MRR, Hits@N</td><td>Link Prediction</td></tr> <tr><td>Querying &amp; Reasoning</td><td>Forecasting &amp; Reasoning</td><td><b>TG-LLM</b></td><td>TGQA, TimeQA, TempReason</td><td>F1, Accuracy, Exact Match</td><td>Temporal Reasoning</td></tr> <tr><td>Querying &amp; Reasoning</td><td>Forecasting &amp; Reasoning</td><td><b>LLM4DyG</b></td><td>Enron, DBLP, Flights</td><td>Accuracy, F1, Recall</td><td>Spatio-Temporal Reasoning, Graph Reasoning &amp; Querying, Link Prediction</td></tr> <tr><td>Querying &amp; Reasoning</td><td>QA &amp; Interpretability</td><td><b>TimeR<sup>4</sup></b></td><td>MULTITQ, TimeQuestions</td><td>Hits@N</td><td>Temporal KGQA</td></tr> <tr><td>Querying &amp; Reasoning</td><td>QA &amp; Interpretability</td><td><b>GenTKGQA</b></td><td>CronQuestion, TimeQuestions</td><td>Hits@N</td><td>Temporal KGQA</td></tr> <tr><td>Querying &amp; Reasoning</td><td>QA &amp; Interpretability</td><td><b>Unveiling LLMs</b></td><td>FEVER, CLIMATE-FEVER</td><td>Precision, Recall, F1, ROC AUC, Accuracy</td><td>Claim Verification</td></tr>
<tr><td>Generating &amp; Updating</td><td>Generating Structures</td><td><b>FinDKG</b></td><td>WIKI, YAGO, ICEWS14</td><td>MRR, Hits@N</td><td>Link Prediction</td></tr>
<tr><td>Generating &amp; Updating</td><td>Generating Structures</td><td><b>GenTKG</b></td><td>ICEWS14, ICEWS18, GDELT, YAGO</td><td>Hits@N</td><td>Link Prediction</td></tr>
<tr><td>Generating &amp; Updating</td><td>Generating Structures</td><td><b>Up To Date</b></td><td>Wikidata</td><td>Accuracy, Response Rate</td><td>Fact Validation, QA</td></tr>
<tr><td>Generating &amp; Updating</td><td>Generating Structures</td><td><b>PPT</b></td><td>ICEWS14, ICEWS18, ICEWS05-15</td><td>MRR, Hits@N</td><td>Link Prediction</td></tr>
<tr><td>Generating &amp; Updating</td><td>Generating Structures</td><td><b>LLM-DA</b></td><td>ICEWS14, ICEWS05-15</td><td>MRR, Hits@N</td><td>Link Prediction</td></tr>
<tr><td>Generating &amp; Updating</td><td>Generating Insights &amp; Representations</td><td><b>TimeLlama</b></td><td>ICEWS14, ICEWS18, ICEWS05-15</td><td>Precision, Recall, F1, BLEU, ROUGE</td><td>Event Forecasting, Explanation Generation</td></tr>
<tr><td>Generating &amp; Updating</td><td>Generating Insights &amp; Representations</td><td><b>RealTCD</b></td><td>Simulation Datasets</td><td>SHD, SID</td><td>Temporal Causal Discovery, Anomaly Detection</td></tr>
<tr><td>Generating &amp; Updating</td><td>Generating Insights &amp; Representations</td><td><b>DynLLM</b></td><td>Tmall, Alibaba</td><td>Recall@K, NDCG@K</td><td>Dynamic Graph Recommendation, Top-K Recommendation</td></tr>

<tr><td>Evaluation &amp; Application</td><td>Model Evaluation</td><td><b>Dynamic-TempLAMA</b></td><td>DYNAMICTEMPLAMA</td><td>Accuracy, MRR, ROUGE, F1</td><td>Temporal Robustness Evaluation, Factual Knowledge Probing</td></tr>
<tr><td>Evaluation &amp; Application</td><td>Model Evaluation</td><td><b>DARG</b></td><td>GSM8K, BBQ, BBH Navigate, BBH Dyck Language</td><td>Accuracy, CIAR, Exact Match, Accuracy</td><td>Mathematical, Social, Spatial, Symbolic Reasoning</td></tr>
<tr><td>Evaluation &amp; Application</td><td>Downstream Applications</td><td><b>AnomalyLLM</b></td><td>UCI Messages, Blogcatalog</td><td>AUC</td><td>Anomaly Detection</td></tr>
<tr><td>Evaluation &amp; Application</td><td>Downstream Applications</td><td><b>MoMa-LLM</b></td><td>iGibson scenes</td><td>AUC, Recall</td><td>Semantic Interactive Object Search</td></tr>
<tr><td>Evaluation &amp; Application</td><td>Downstream Applications</td><td><b>TRR</b></td><td>Reuters Financial News</td><td>AUROC</td><td>Event Detection</td></tr>

</tbody> </table> </div>

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
