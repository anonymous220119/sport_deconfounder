#!/bin/bash

for s in '2016' '2014' '2015' ;do 
    for l in 21518 10257 7809 1729;do 
	for k in {1..10};do
	    python factor_model_validation.py $l $k $s;
	done;
    done;
done;

for s in '2019' ;do 
    for l in 37 49;do 
	for k in {1..10};do
	    python factor_model_validation.py $l $k $s;
	done;
    done;
done;
