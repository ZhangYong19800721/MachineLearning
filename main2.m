clear all;
close all;
rng('shuffle');

D = 10; N = 1e6; S = 100; M = N/S;
MU = 1:D; SIGMA = 10*rand(D); SIGMA = SIGMA * SIGMA';
data = mvnrnd(MU,SIGMA,N)';

% data(1,:) = repmat(5.5,1,N);

[W,D,A] = learn.whiten(data);

whiten_data = W * (data - repmat(A,1,N));
dewhiten_data = D * whiten_data + repmat(A,1,N);

error = sum(sum(abs(dewhiten_data - data))) / N;
