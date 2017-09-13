clear all
close all

F = F();
F = learn.optimal.LINE(F,-0.5,1);
[a,b] = learn.optimal.ARR(F,0,1,1e-20);