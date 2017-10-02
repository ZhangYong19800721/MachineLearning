clear all
close all
rng(1)

load('tiny_images.mat')
data = double(points); data = data / 255; 
clear points;
[D,S,M] = size(data); N = S * M;
S = 100; M = N/S;
data = reshape(data,D,S,M);

load('ps.mat');
% data = reshape(data,D,[]);
% rebuild_data = ps.compute(data);
% finetune_error1 = sum(sum((rebuild_data - data).^2)) / N;
% disp(sprintf('finetune_error1:%f',finetune_error1));
% data = reshape(data,D,S,M);
parameters.algorithm = 'SADAM';
parameters.learn_rate = 1e-3;
parameters.window = 2e4;
parameters.decay = 4;
ps = ps.train(data,data,parameters);

data = reshape(data,D,[]);
rebuild_data = ps.compute(data);
finetune_error2 = sum(sum((rebuild_data - data).^2)) / N;
disp(sprintf('finetune_error2:%f',finetune_error2));
save('ps2.mat','ps');