clear all;
close all;

load('nucleotides.mat');
[D,N] = size(nucleotide);
S = 100; M = floor(N / S); N = S*M;
nucleotide = nucleotide(:,1:N);
nucleotide(nucleotide>0) = 1;
nucleotide = reshape(nucleotide,D,S,M);

rbm = learn.RBM(D,1024);
rbm = rbm.initialize(nucleotide);

parameters.learn_rate = [1e-8,1e-2];
parameters.weight_cost = 1e-4;
parameters.max_it = 1e6;

rbm = rbm.pretrain(nucleotide,parameters);
nucleotide = reshape(nucleotide,D,[]);
recon_nucleotide = rbm.reconstruct(nucleotide);

recon_nucleotide = round(recon_nucleotide); 
recon_nucleotide(recon_nucleotide<0) = 0;

error = sum(sum((nucleotide - recon_nucleotide).^2)) / N;

save;
