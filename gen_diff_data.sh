#!/bin/bash
# Loop 100 times

for ((i=0; i<=153; i++))
do
    # Call the Python script with the call number as argument
    #this is script, step $#, DIffusion steps, guidance
    python3 ablation_gen.py "$i" 30 3
done


