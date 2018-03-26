#1bin/bash

dir=('experiments/comparisons/activations_and_lrs')

for i in "${dir[@]}"
do
    while [[ $(squeue | grep s1769454 | wc -l) -gt 0 ]]
    do
        sleep 1
    done
    
    cd $i
    sbatch script.sh
    cd ~/self_driving_game
done
