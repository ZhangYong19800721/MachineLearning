clear all;
close all;
[data,~,~,~] = learn.import_mnist('./+learn/mnist.mat');
[D,S,M] = size(data); N = S * M;

configure = [D,500,256];
sae = learn.SAE(configure);

parameters.learn_rate = [1e-6,1e-2];
parameters.weight_cost = 1e-4;
parameters.max_it = 1e6;
sae = sae.pretrain(data,parameters);
save('sae_mnist_pretrain.mat','sae');
% load('sae_mnist_pretrain.mat');

data = reshape(data,D,[]);
recon_data1 = sae.rebuild(data);
error1 = sum(sum((round(255*recon_data1) - 255*data).^2)) / N;

data = reshape(data,D,S,M);
parameters.max_it = 1e6;
sae = sae.train(data,parameters);

save('sae_mnist_finetune.mat','sae');

data = reshape(data,D,[]);
recon_data2 = sae.rebuild(data);
error2 = sum(sum((round(255*recon_data2) - 255*data).^2)) / N;

