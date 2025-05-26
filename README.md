# MBIDR
## Example to run the codes
This repository is an official PyTorch implementation of MBIDR in "Multi-Behavior Intent Disentangled Learning for Fine-Grained Interest Discovery in Recommendation".

### Taobao
python MBIDR.py --dataset Taobao --wid [0.01,0.01,0.01] --coefficient [1.0/6,5.0/6,0.0/6] --decay 0.01 --batch_size 512

### Beibei
python MBIDR.py --dataset Beibei --wid [0.1,0.1,0.1] --coefficient [0.0/6,6.0/6,0.0/6] --decay 10 --batch_size 512
