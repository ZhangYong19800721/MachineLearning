clear all
close all

F = F();
F = learn.optimal.LINE(F,-0.5,1);
[a,b] = learn.optimal.AR(F,0,1);