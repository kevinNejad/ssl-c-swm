#!/bin/bash
echo "Training"
for S in 111 222 333 444 555
do
    for L in contrastive vic 
    do
    python train.py --dataset data/shapes_train.h5 --encoder small --name shapes --epochs 150 --seed $S --ssl-loss $L
    done
done

echo "Evaluation"

for S in 111 222 333 444 555
do
    for L in contrastive vic 
    do
        for T in 1 5 10
        do
            python eval.py --dataset data/shapes_eval.h5 --save-folder checkpoints/$S/shapes_$L --num-steps $T
        done
    done
done

