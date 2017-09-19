clear all
close all

[data,~,~,~] = learn.data.import_mnist('./+learn/+data/mnist.mat');
data = reshape(data,784,[]);
whiten = learn.tools.whiten();
whiten = whiten.pca(data);

w_data = whiten.white(data);
d_data = whiten.dewhite(w_data);

