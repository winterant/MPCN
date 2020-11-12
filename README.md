MPCN
===
> Implementation for the paper：  
Tay, Yi, Anh Tuan Luu, and Siu Cheung Hui. "Multi-pointer co-attention networks for recommendation." In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 2309-2318. 2018.

# Environments
  + python 3.8
  + pytorch 1.60

# Dataset
  You need to prepare the following documents:  
  1. dataset(`/data/music/Digital_Music.json`)  
   Download from http://deepyeti.ucsd.edu/jianmo/amazon/index.html (Choose Digital Music)

# Running

Process dataset
```
python preprocess.py
```

Train and evaluate the model:
```
python main.py
```
