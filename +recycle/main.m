clear all;
close all;
rng(1);

[data,~,~,~] = learn.data.import_mnist('./+learn/+data/mnist.mat');
[D,S,M] = size(data); N = S * M;

configure = [D,500,500,2000,2];
sae = learn.neural.SAE(configure);

parameters.learn_rate = [1e-6,1e-2];
parameters.weight_cost = 1e-4;
parameters.max_it = 1e0;
sae = sae.pretrain(data,parameters);
% save('sae_mnist_pretrain.mat','sae');
load('sae_mnist_pretrain.mat');
sae = sae.weightsync();

data = reshape(data,D,[]);
recon_data = sae.rebuild(data,'nosample');
error = sum(sum((recon_data - data).^2)) / N;
disp(sprintf('pretrain-error:%f',error));

configure = [D,500,500,2000,2,2000,500,500,D];
ps = learn.neural.PerceptionS(configure);
ps.weight = [
    reshape(sae.rbms{1}.weight_v2h,[],1); % Dx500
    reshape(sae.rbms{1}.hidden_bias,[],1); % 500
    reshape(sae.rbms{2}.weight_v2h,[],1); % 500x500
    reshape(sae.rbms{2}.hidden_bias,[],1); % 500
    reshape(sae.rbms{3}.weight_v2h,[],1); % 500x2000
    reshape(sae.rbms{3}.hidden_bias,[],1); % 2000
    reshape(sae.rbms{4}.weight_v2h,[],1); % 2000x2
    reshape(sae.rbms{4}.hidden_bias,[],1); % 2
    reshape(sae.rbms{4}.weight_h2v',[],1); % 2x2000
    reshape(sae.rbms{4}.visual_bias,[],1); % 2000
    reshape(sae.rbms{3}.weight_h2v',[],1); % 2000x500
    reshape(sae.rbms{3}.visual_bias,[],1); % 500
    reshape(sae.rbms{2}.weight_h2v',[],1); % 500x500
    reshape(sae.rbms{2}.visual_bias,[],1); % 500
    reshape(sae.rbms{1}.weight_h2v',[],1); % 500xD
    reshape(sae.rbms{1}.visual_bias,[],1)]; % D

recon_data = ps.do(data);
error = sum(sum((recon_data - data).^2)) / N;
disp(sprintf('pretrain-error:%f',error));

cgbps = learn.neural.CGBPS(data,data,ps);
clear parameters;
parameters.epsilon = 1e-3; % 当梯度模小于epsilon时停止迭代
parameters.alfa = 1e2;     % 线性搜索区间倍数
parameters.beda = 1e-4;    % 线性搜索的停止条件
parameters.max_it = 1e4;   % 最大迭代次数
parameters.reset = 500;    % 重置条件
weight = learn.optimal.minimize_cg(cgbps,ps.weight,parameters);
ps.weight = weight;

recon_data = ps.do(data);
error = sum(sum((recon_data - data).^2)) / N;
disp(sprintf('pretrain-error:%f',error));