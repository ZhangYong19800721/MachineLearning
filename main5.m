clear all
close all
rng(1)

load('tiny_images.mat')
data = double(points); data = data / 255;
[D,S,M] = size(data); N = S * M;

configure = [D,1024,1024,64];
sae = learn.neural.SAE(configure);

parameters.learn_rate = 1e-1;
parameters.max_it = 0; % M*100;
parameters.decay = 1000;
sae = sae.pretrain(data,parameters);
% save('sae_pretrained.mat','sae');
load('sae_pretrained.mat');

data = reshape(data,D,[]);
recon_data = sae.rebuild(data,'nosample');
pretrain_error = sum(sum((recon_data - data).^2)) / N;
disp(sprintf('pretrained-error:%f',pretrain_error));

data = reshape(data,D,S,M);
clear parameters;
parameters.learn_rate_max = 1e-1;
parameters.learn_rate_min = 1e-5;
parameters.momentum = 0.9;
parameters.max_it = 0; % M*100;
parameters.case = 2; % 无抽样的情况
sae = sae.train(data,parameters);
%save('sae_trained.mat','sae');
load('sae_trained.mat');

data = reshape(data,D,[]);
recon_data = sae.rebuild(data,'nosample');
train_error = sum(sum((recon_data - data).^2)) / N;
disp(sprintf('trained-error:%f',train_error));