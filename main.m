clear all;
close all;

load('nucleotides.mat');

[D,N] = size(nucleotide);
shuffle_idx = randperm(N);
nucleotide = nucleotide(:,shuffle_idx);
S = 100; M = floor(N / S); N = S*M;
nucleotide = nucleotide(:,1:N);

[W,R,A] = learn.whiten(nucleotide);
data = W * (nucleotide - repmat(A,1,size(nucleotide,2)));

ave_value = mean(data,2);
std_value = std(data,0,2);

data = reshape(data,D,S,M);

gbrbm = learn.GBRBM(D,1024);
gbrbm = gbrbm.initialize(data);

parameters.learn_rate = [1e-8,1e-4];
parameters.learn_sgma = 1e-2;
parameters.weight_cost = 1e-4;
parameters.max_it = 1e6;

gbrbm = gbrbm.pretrain(data,parameters);
recon_data = gbrbm.reconstruct(data);
recon_nucleotide = R * recon_data + repmat(A,1,size(recon_data,2));

nucleotide = uint8(nucleotide);
recon_nucleotide = uint8(recon_nucleotide);

error = sum(sum((nucleotide - recon_nucleotide).^2)) / N;

save('gbrbm_nucleotide.mat');
