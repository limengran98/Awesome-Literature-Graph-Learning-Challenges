
# 📚 Awesome Literature: Graph Learning Challenges with LLMs

A curated list of recent research addressing fundamental challenges in graph learning with the assistance of large language models (LLMs). Papers are categorized by **challenge type** and **methodological approach**.

## Table of Contents

- [Incompleteness in Graphs](#incompleteness-in-graphs)
  - [Robust Graph Learning](#robust-graph-learning)
  - [Few-shot Graph Learning](#few-shot-graph-learning)
  - [Knowledge Graph Completion](#knowledge-graph-completion)
- [Imbalance in Graphs](#imbalance-in-graphs)
  - [Class-Imbalanced Graph Learning](#class-imbalanced-graph-learning)
  - [Structure-Imbalanced Graph Learning](#structure-imbalanced-graph-learning)
- [Cross-Domain Heterogeneity in Graphs](#cross-domain-heterogeneity-in-graphs)
  - [Text-Attributed Graph Learning](#text-attributed-graph-learning)
  - [Multimodal Attributed Graph Learning](#multimodal-attributed-graph-learning)
  - [Structural Heterogeneous Graph Learning](#structural-heterogeneous-graph-learning)
- [Dynamic Instability in Graphs](#dynamic-instability-in-graphs)
  - [LLMs for Querying and Reasoning](#llms-for-querying-and-reasoning)
  - [LLMs for Generating and Updating](#llms-for-generating-and-updating)
  - [LLMs for Evaluation and Application](#llms-for-evaluation-and-application)

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


