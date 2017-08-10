clear all;
close all;

load('nucleotides.mat');

[D,N] = size(nucleotide);
shuffle_idx = randperm(N);
nucleotide = nucleotide(:,shuffle_idx);
S = 100; M = floor(N / S); N = S*M;
nucleotide = nucleotide(:,1:N);

nucleotide = reshape(nucleotide,D,S,M);

gbrbm = learn.GBRBM(D,1024);
gbrbm = gbrbm.initialize(nucleotide);

parameters.learn_rate = [1e-10,1e-3];
parameters.learn_sgma = 1e-2;
parameters.weight_cost = 1e-4;
parameters.max_it = 1e6;

gbrbm = gbrbm.pretrain(nucleotide,parameters);
nucleotide = reshape(nucleotide,D,[]);
recon_nucleotide = gbrbm.reconstruct(nucleotide);

nucleotide = round(nucleotide);
recon_nucleotide = round(recon_nucleotide); 
recon_nucleotide(recon_nucleotide<0) = 0;

error = sum(sum((nucleotide - recon_nucleotide).^2)) / N;

save('gbrbm_nucleotide.mat');
