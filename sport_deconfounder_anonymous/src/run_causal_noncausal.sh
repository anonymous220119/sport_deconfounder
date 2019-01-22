#!/bin/bash

m=8
for s in '2016' '2015' '2014' ;do 
    for l in 21518 10257 7809 1729;do 
	for k in {1 5 10 };do
	    for t in 0 1 2; do # split_type
		python causal_noncausal_validation.py $l $s $t $k $m ;
	    done;
	done;
    done;
done;
 

