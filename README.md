# kaggle_LECR

# [Learning Equality - Curriculum Recommendations](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations)  
> The goal of this competition is to streamline the process of matching educational content to specific topics in a curriculum. You will develop an accurate and efficient model trained on a library of K-12 educational materials that have been organized into a variety of topic taxonomies. These materials are in diverse languages, and cover a wide range of topics, particularly in STEM (Science, Technology, Engineering, and Mathematics).

# Solution Overview

## CV setting
GroupKFold(n_splits=3).split(groups=df["channel"]) with category non-source topic.  
If multiple folds are used, there will be multiple retrieval results (1st_stage_model) for inference, and I could not think of a good way to integrate them, so I used only one fold.
## 1st stage model(Retriever)
Extract candidates from a large amount of content based on the cosine similarity of sentence embedding.
* DataLoader: [NoDuplicatesDataLoader](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/datasets/NoDuplicatesDataLoader.py)
* Loss: [MultipleNegativesSymmetricRankingLoss](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesSymmetricRankingLoss.py)
* Evaluator: [InformationRetrievalEvaluator](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/InformationRetrievalEvaluator.py)
    * class inheritance(score: MAP@k -> recall@k)
* input sentence
    * content: title + [SEP] + description
    * topic: title + [SEP] + description + [SEP] + context
* A100, 40GB(colab pro+ premium GPU)
    * batch_size is the most important parameter when using MNSLoss
        
| filename | model | batch_size | max_len | Recall@50(fold0) | Memory(/40GB) |
| - | - | - | - | - | - |
| exp004 | all-MiniLM-L6-v2 | 768 | 128 | 0.8216 | 34GB |
| exp006 | all-mpnet-base-v2 | 256 | 128 | 0.8502 | 37.3GB |
| exp007 | xlm-roberta-base | 224 | 128 | 0.8421 | 36.4GB |
| exp008 | paraphrase-multilingual-mpnet-base-v2 | 224 | 128 | 0.8207 | 36.4GB |


## 2nd stage model(Re-ranker)
Determine whether pairs of extracted candidates are correctly paired.
* Using all-mpnet-base-v2 top_k:50 pairs and all positive examples
* input sentence 
    * content: title + [SEP] + description
    * topic: title + [SEP] + description + [SEP] + context
    * Using tokenizer's text_pair allows both texts to be entered in a balanced manner.
* Search for thresholds & top_k.
  * I want to assign at least one content to each topic, so I assign k content of pairs of topics that were not assigned and have a large probability (which I also search for)
    
| filename | model | batch_size | max_len | f2_score<br>(fold0) | add topk<br>(fold0) | thres<br>(fold0) | best_epoch |
| - | - | - | - | - | - | - | - |
| exp004 | all-MiniLM-L6-v2 | 96 | 256 | 0.5004 | 12 | 0.051 | 4 |
| exp006 | all-mpnet-base-v2 | 32 | 256 | 0.5591 | 14 | 0.001 | 4 | 
| exp007 | xlm-roberta-base | 32 | 256 | 0.5630 | 18 | 0.00018 | 4 |
| exp008 | paraphrase-multilingual-mpnet-base-v2 | 32 | 256 | 0.5456 | 10 | 0.00049 | 4 |


## 3rd stage model(Weighted average)
Ensemble to improve score.
* Find the weights that minimize BCEWithLogitsLoss using optuna
* Search for thresholds & top_k.
  * I want to assign at least one content to each topic, so I assign k content of pairs of topics that were not assigned and have a large probability (which I also search for)
  
## 3rd stage model(stacking lgb)
Stacking using LightGBM.
* metric: cross_entropy
* useful features: Cosine Similarity of two sentences(from 1st_stage_model), logits(from 2nd_stage_model)
* Search for thresholds & top_k.
  * I want to assign at least one content to each topic, so I assign k content of pairs of topics that were not assigned and have a large probability (which I also search for)

# Published notebook
https://www.kaggle.com/code/yujikomi/lecr-preprocessing-considering-language  
This notebook is based on the idea that the language and content of the topic is almost the same.   I have shown that recall can be improved by using this.

# CV vs LB vs PB

| file | CV | LB | PB |
| - | - | - | - |
| submission/weighted_average_ver1 | 0.5937 | 0.55654 | 0.5859 |
| submission/weighted_average_ver2 | 0.5955 | 0.55752 | 0.58541 |
| submission/stacking_lgb | 0.6025 | 0.56759 | 0.59552 |
| exp004 | 0.5004 | 0.46699 | 0.48793 |
| exp006 | 0.5591 | 0.51199 | 0.54092 |
| exp007 | 0.5630 | 0.5301 | 0.56182 |
| exp008 | 0.5456 | 0.51697 | 0.54745 |


# Execution environments
kaggle notebook, colab pro+

# Ranking
* LB: 75th
* PB: 73th (bronze medal)
