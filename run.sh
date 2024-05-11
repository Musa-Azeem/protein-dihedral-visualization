#!/bin/bash
for i in {5..25..5}
do
    for j in 0.1 none
    do
        python3 script.py 4 5 $i $j
        python3 script.py 4 6 $i $j
        python3 script.py 5 6 $i $j
    done
done