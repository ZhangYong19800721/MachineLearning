clear all;
close all;
rng(1);

[data,~,~,~] = learn.import_mnist('./+learn/mnist.mat'); 
[D,S,M] = size(data); 
batchs = zeros(S,D,M);

for m = 1:M
    batchs(:,:,m) = data(:,:,m)';
end

batchs = batchs * 255;

clear data;

parameters.v_var = 1;
parameters.epsilonw_vng = 0.01;
parameters.std_rate = 1000;
parameters.maxepoch  = 1000;
parameters.PreWts.vhW = randn(D,500);
parameters.PreWts.hb  = zeros(1,500);
parameters.PreWts.vb  = zeros(1,D);
parameters.nHidNodes  = 500;
parameters.nCD = 1;
parameters.init_final_momen_iter = 60;
parameters.init_momen = 0.5;
parameters.final_momen = 0.9;
parameters.wtcost = 0;
parameters.SPARSE = 0;

[weight,visual_bias,hidden_bias,fvar,errs] = learn.GaussianRBM(batchs,parameters);