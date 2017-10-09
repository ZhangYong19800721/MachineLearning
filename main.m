clear all;
close all;
rng(1);

load('./+learn/+data/mnist.mat');
points = mnist_train_images;
[D,N] = size(points); S = 100; M = N/S;
points = reshape(points,D,S,M); 
points = double(points) / 255.0;
data = reshape(points,D,[]);

configure = [D,500,64];
sae = learn.neural.SAE(configure);

parameters.learn_rate = 1e-1;
parameters.max_it = M*100;
parameters.decay = 1000;
parameters.window = 1e4;
sae = sae.pretrain(points,parameters);
rebuild_data = sae.rebuild(data,'nosample');
pretrain_error = sum(sum((rebuild_data - data).^2)) / N;
disp(sprintf('pretrain_error:%f',pretrain_error));
save('sae_p.mat','sae');

clear parameters;
parameters.learn_rate_max = 1e-1;
parameters.learn_rate_min = 1e-6;
parameters.momentum = 0.9;
parameters.max_it = 1e6;
parameters.case = 2; % 无抽样的情况
sae = sae.train(points,parameters);
rebuild_data = sae.rebuild(data,'nosample');
train_error1 = sum(sum((rebuild_data - data).^2)) / N;
disp(sprintf('train_error1:%f',train_error1));
save('sae_t.mat','sae');

configure = [D,500,64,500,D];
ps = learn.neural.PerceptionS(configure); L = 2;
weight = [];
for l = 1:L
    weight = [weight; reshape(sae.rbms{l}.weight_v2h, [],1); reshape(sae.rbms{l}.hidden_bias,[],1)];
end
for l = L:-1:1
    weight = [weight; reshape(sae.rbms{l}.weight_h2v', [],1); reshape(sae.rbms{l}.visual_bias,[],1)];
end
ps.weight = weight;

rebuild_data = ps.compute(data);
train_error2 = sum(sum((rebuild_data - data).^2)) / N;
disp(sprintf('train-error2:%f',train_error2));

clear parameters;
parameters.algorithm = 'SADAM';
parameters.learn_rate = 1e-3;
parameters.window = 1e4;
parameters.decay = 8;
ps = ps.train(points,points,parameters);

rebuild_data = ps.compute(data);
finetune_error = sum(sum((rebuild_data - data).^2)) / N;
disp(sprintf('finetune_error:%f',finetune_error));
save('ps.mat','ps');