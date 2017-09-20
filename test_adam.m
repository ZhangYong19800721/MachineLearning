clear all
close all

x0 = [0 0]';
F = F();
x = learn.optimal.minimize_adam(F,x0);