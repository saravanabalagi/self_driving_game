#1bin/bash

dir=('experiments/comparisons/batch_norm')

for i in "${dir[@]}"
do
    while [[ $(squeue | grep $(whoami) | wc -l) -gt 0 ]]
    do
        sleep 1
    done
    
    cd $i
    sbatch script.sh
    cd ~/self_driving_game
done
