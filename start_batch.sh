#1bin/bash

dir=('experiments/regression/frame_map')

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
