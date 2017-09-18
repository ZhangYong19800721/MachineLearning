clear all;
close all;

%% 载入数据
load('images.mat');
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
parameters.max_it = 1e6;
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
parameters.max_it = 1e6;
sae = sae.train(points,parameters);
if parameters.max_it > 1
    save('sae_train_1024x64.mat','sae');
end
points = reshape(points,D,[]);
rebuild_points = sae.rebuild(points,'nosample');
error_trained1 = sum(sum((rebuild_points - points).^2)) / N;
disp(sprintf('train-error1:%f',error_trained1));

%% 调优训练
configure = [D,64,D];
ps = learn.neural.PerceptionS(configure);
ps.weight = [
    reshape(sae.rbms{1}.weight_v2h,[],1); % Dx64
    reshape(sae.rbms{1}.hidden_bias,[],1); % 64
    reshape(sae.rbms{1}.weight_h2v',[],1); % 64xD
    reshape(sae.rbms{1}.visual_bias,[],1)  % D
    ];

recon_points = ps.do(points);
error_trained2 = sum(sum((recon_points - points).^2)) / N;
disp(sprintf('train-error2:%f',error_trained2));

cgbps = learn.neural.CGBPS(points,points,ps);
clear parameters;
parameters.epsilon = 1e-3; % 当梯度模小于epsilon时停止迭代
weight = learn.optimal.minimize_cg(cgbps,ps.weight,parameters);
ps.weight = weight;
save('perception.mat','ps');
load('perception.mat');

recon_points = ps.do(points);
error_finetune = sum(sum((recon_points - points).^2)) / N;
disp(sprintf('finetune-error:%f',error_finetune));