clear all;
close all;
rng(1);

D = 10; N = 1e5; S = 100; M = 1000;
MU = 1:D; SIGMA = 10*rand(D); SIGMA = SIGMA * SIGMA';
data = mvnrnd(MU,SIGMA,N)';
X = data; AVE_X = repmat(mean(X,2),1,N);
Z = double(X) - AVE_X;
Y = Z*Z';
[P,ZK] = eig(Y);
ZK=diag(ZK);
ZK(ZK<=0)=0;
DK=ZK; DK(ZK>0)=1./(ZK(ZK>0));

trwhitening =    sqrt(N-1)  * P * diag(sqrt(DK)) * P';
dewhitening = (1/sqrt(N-1)) * P * diag(sqrt(ZK)) * P';

data = trwhitening * Z;
data = reshape(data,D,S,M);

for minibatch_idx = 1:M
    mnist(:,:,minibatch_idx) = data(:,:,minibatch_idx)';
end

parameters.v_var = 1;
parameters.epsilonw_vng = 1e-3;
parameters.std_rate = 1000;
parameters.max_it  = 1e6;
parameters.PreWts.vhW = 0.01*randn(D,100);
parameters.PreWts.hb  = zeros(1,100);
parameters.PreWts.vb  = zeros(1,D);
parameters.nHidNodes  = 100;
parameters.nCD = 1;
parameters.init_final_momen_iter = 60;
parameters.init_momen = 0.5;
parameters.final_momen = 0.9;
parameters.wtcost = 0;
parameters.SPARSE = 0;

[weight,visual_bias,hidden_bias] = learn.GaussianRBM(mnist,parameters);