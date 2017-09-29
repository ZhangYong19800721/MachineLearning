clear all
close all
rng(1)

load('tiny_images.mat')
data = double(points); data = data / 255;
[D,S,M] = size(data); N = S * M;
load('sae_trained.mat');

data = reshape(data,D,[]);
recon_data = sae.rebuild(data,'nosample');
train_error = sum(sum((recon_data - data).^2)) / N;
disp(sprintf('trained-error:%f',train_error));