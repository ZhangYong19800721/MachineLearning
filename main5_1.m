clear all
close all
rng(1)

load('tiny_images.mat')
data = double(points); data = data / 255;
[D,S,M] = size(data); N = S * M;
S = 100; M = N/S;
data = reshape(data,D,S,M);
load('sae_trained.mat');

configure = [D,1024,1024,64,1024,1024,D]; L = 3;
ps = learn.neural.PerceptionS(configure);
weight = [];
for l = 1:L
    weight = [weight; reshape(sae.rbms{l}.weight_v2h, [],1); reshape(sae.rbms{l}.hidden_bias,[],1)];
end

for l = L:-1:1
    weight = [weight; reshape(sae.rbms{l}.weight_h2v', [],1); reshape(sae.rbms{l}.visual_bias,[],1)];
end

ps.weight = weight;

parameters.algorithm = 'ADAM';
parameters.learn_rate = 1e-3;
parameters.window = 1e4;
parameters.decay = 6;
ps = ps.train(data,data,parameters);

data = reshape(data,D,[]);
recon_data = ps.compute(data);
train_error = sum(sum((recon_data - data).^2)) / N;
disp(sprintf('finetune-error:%f',train_error));
save('ps.mat','ps');