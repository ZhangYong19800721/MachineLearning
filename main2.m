clear all
close all

F = F();
x = [-1 1]';
d = [1 -2]';

parameters.armijo.alfa = 0.2;
parameters.armijo.beda = 0.5;
parameters.armijo.maxs = 30;

[lamda,nf,nx] = learn.optimal.armijo(F,x,F.gradient(x),d,parameters);