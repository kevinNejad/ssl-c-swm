# Self-supervised Learning of Structured World Models

This repository is the code for comparing the performance of contrastive vs regulariser (nonb-contrastive) self-supervised [learning of structure world models](http://arxiv.org/abs/1911.12247)(Thomas Kipf, Elise van der Pol, Max Welling).

This is based on the [original implementation of C-SWM](https://github.com/tkipf/c-swm). The only filed that is modified is `modules.py`. 

## Requirements

* Python 3.6 or 3.7
* PyTorch version 1.2
* OpenAI Gym version: 0.12.0 `pip install gym==0.12.0`
* OpenAI Atari_py version: 0.1.4: `pip install atari-py==0.1.4`
* Scikit-image version 0.15.0 `pip install scikit-image==0.15.0`
* Matplotlib version 3.0.2 `pip install matplotlib==3.0.2`

## Generate datasets

**2D Shapes**:
```bash
python data_gen/env.py --env_id ShapesTrain-v0 --fname data/shapes_train.h5 --num_episodes 1000 --seed 1
python data_gen/env.py --env_id ShapesEval-v0 --fname data/shapes_eval.h5 --num_episodes 10000 --seed 2
```


**Atari Pong**:
```bash
python data_gen/env.py --env_id PongDeterministic-v4 --fname data/pong_train.h5 --num_episodes 1000 --atari --seed 1
python data_gen/env.py --env_id PongDeterministic-v4 --fname data/pong_eval.h5 --num_episodes 100 --atari --seed 2
```


## Run model training and evaluation
You need to pass the type of self-supervised loss function as an argument. Currently, the options are `contrastive` and `vic`.

**2D Shapes**:
```bash
python train.py --dataset data/shapes_train.h5 --encoder small --name shapes --ssl-loss vic
python eval.py --dataset data/shapes_eval.h5 --save-folder checkpoints/shapes_vic --num-steps 1
```

**Atari Pong**:
```bash
python train.py --dataset data/pong_train.h5 --encoder medium --embedding-dim 4 --action-dim 6 --num-objects 3 --copy-action --epochs 200 --name pong --ssl-loss vic
python eval.py --dataset data/pong_eval.h5 --save-folder checkpoints/pong_vic --num-steps 1
```

## Results

### 2D Shapes
| Loss        | H@1            | MRR          |
|-------------|----------------|--------------|
| contrastive | **99** $\pm$ 0.0  | **99** $\pm$ 0.0 |
| VICreg      | **99** $\pm$ 0.0 | **99** $\pm$ 0.0 |

### Pong
| Loss        | H@1            | MRR          |
|-------------|----------------|--------------|
| contrastive | 39 $\pm$ 14.6   | 57 $\pm$ 11.3 |
| VICreg      | **46.8** $\pm$ 11.9 | **62** $\pm$ 10.9 |
