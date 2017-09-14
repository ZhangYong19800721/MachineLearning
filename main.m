clear all;
close all;

load('images.mat');
N = 73600;
points = points(:,1:N);

D = 3072; S = 100; M = N/S;
points = reshape(points,D,S,M);
points = double(points) / 255;

configure = [D,1024,128];
sae = learn.neural.SAE(configure);

parameters.learn_rate = [1e-6,1e-2];
parameters.weight_cost = 1e-4;
parameters.max_it = 1e6;
sae = sae.pretrain(points,parameters);
save('sae_pretrain.mat','sae');
load('sae_pretrain.mat');
sae = sae.weightsync();

points = reshape(points,D,[]);
rebuild_points = sae.rebuild(points,'nosample');
error = sum(sum((rebuild_points - points).^2)) / N;
disp(sprintf('pretrain-error:%f',error));

configure = [D,1024,128,1024,D];
ps = learn.neural.PerceptionS(configure);
ps.weight = [
    reshape(sae.rbms{1}.weight_v2h,[],1); % Dx1024
    reshape(sae.rbms{1}.hidden_bias,[],1); % 1024
    reshape(sae.rbms{2}.weight_v2h,[],1); % 1024x128
    reshape(sae.rbms{2}.hidden_bias,[],1); % 128
    reshape(sae.rbms{2}.weight_h2v',[],1); % 128x1024
    reshape(sae.rbms{2}.visual_bias,[],1); % 1024
    reshape(sae.rbms{1}.weight_h2v',[],1); % 1024xD
    reshape(sae.rbms{1}.visual_bias,[],1)  % D
    ]; % D

recon_points = ps.do(points);
error = sum(sum((recon_points - points).^2)) / N;
disp(sprintf('pretrain-error:%f',error));

cgbps = learn.neural.CGBPS(points,points,ps);
clear parameters;
parameters.epsilon = 1e-3; % 当梯度模小于epsilon时停止迭代
parameters.max_it = 1e5;   % 最大迭代次数
parameters.reset = 500;    % 重置条件
parameters.option = 'gold';% 黄金分割法（精确搜索）
parameters.gold.epsilon = 1e-5; % 线性搜索的停止条件
weight = learn.optimal.minimize_cg(cgbps,ps.weight,parameters);
ps.weight = weight;
save('perception.mat','ps');
load('perception.mat');

recon_points = ps.do(points);
error = sum(sum((recon_points - points).^2)) / N;
disp(sprintf('finetune-error:%f',error));