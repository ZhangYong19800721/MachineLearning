clear all
close all

F = F();
parameters.parabola.epsilon = 1e-20;
[y,x] = learn.optimal.parabola(F,0,2,parameters);