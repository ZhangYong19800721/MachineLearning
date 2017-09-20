clear all;
close all;
rng(3);

%% 载入数据
load('images.mat'); 
load('labels_pos.mat'); labels_pos = labels;
load('labels_neg.mat'); labels = [labels_pos labels_neg];
N = 73600;
points = points(1:(32*32),1:N);
D = 32*32; S = 100; M = N/S;
points = reshape(points,D,S,M);
points = double(points) / 255;

%% 配置自动编码器
configure = [D,64];
sae = learn.neural.SAE(configure);

%% 预训练
parameters.learn_rate = [1e-6,1e-2];
parameters.weight_cost = 1e-4;
parameters.max_it = 1e0;
parameters.case = 2;
sae = sae.pretrain(points,parameters);
if parameters.max_it > 1
    save('sae_pretrain_1024x64.mat','sae');
end
load('sae_pretrain_1024x64.mat');
points = reshape(points,D,[]);
rebuild_points = sae.rebuild(points,'nosample');
error_pretrained = sum(sum((rebuild_points - points).^2)) / N;
disp(sprintf('pretrain-error:%f',error_pretrained));

%% 训练
points = reshape(points,D,S,M);
parameters.max_it = 1e0;
sae = sae.train(points,parameters);
if parameters.max_it > 1
    save('sae_train_1024x64.mat','sae');
end
load('sae_train_1024x64.mat');
points = reshape(points,D,[]);
rebuild_points = sae.rebuild(points,'nosample');
error_trained1 = sum(sum((rebuild_points - points).^2)) / N;
disp(sprintf('train-error1:%f',error_trained1));

%% 调优训练
lnca = learn.ssc.LNCA(D,64);
lnca.weight = [
    reshape(sae.rbms{1}.weight_v2h,[],1); % Dx64
    reshape(sae.rbms{1}.hidden_bias,[],1); % 64
    reshape(sae.rbms{1}.weight_h2v',[],1); % 64xD
    reshape(sae.rbms{1}.visual_bias,[],1)  % D
    ];

recon_points = lnca.do(points);
error_trained2 = sum(sum((recon_points - points).^2)) / N;
disp(sprintf('train-error2:%f',error_trained2));

load('images.mat'); 
points = points(1:(32*32),:); 
points = double(points) / 255; % 重新载入points
lnca_aid = learn.ssc.LNCA_AID(points,labels,lnca);
clear parameters;
parameters.epsilon = 1e-3; % 当梯度模小于epsilon时停止迭代
parameters.max_it = 1e3;
weight = learn.optimal.maximize_cg(lnca_aid,lnca.weight,parameters);
lnca.weight = weight;
save('lnca.mat','lnca');