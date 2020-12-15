MPCN
===
> Implementation for the paper：  
Tay, Yi, Anh Tuan Luu, and Siu Cheung Hui. "Multi-pointer co-attention networks for recommendation." In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 2309-2318. 2018.

# Environments
  + python 3.8
  + pytorch 1.70

# Dataset

You need to prepare the following documents:  
1. Dataset(`data/Digital_Music.json.gz`)  
   Download from http://deepyeti.ucsd.edu/jianmo/amazon/index.html (Choose Digital Music)

2. Word Embedding(`embedding/glove.6B.100d.txt`)  
   Download from https://nlp.stanford.edu/projects/glove

# Pre-Process

Preprocess origin dataset which is json format to be train.csv,valid.csv and test.csv. 
```
python data_process.py
```

# Running

Train and test:
```
python main.py --device cuda:0
```
