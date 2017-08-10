clear all;
close all;

load('nucleotides.mat');

[D,N] = size(nucleotide);
shuffle_idx = randperm(N);
nucleotide = nucleotide(:,shuffle_idx);

S = 100; M = floor(N / S); N = S*M;
nucleotide = nucleotide(:,1:N);
nucleotide = reshape(nucleotide,D,S,M);

% configure = [8688,1024,64];
gbrbm = learn.GBRBM(D,1024);
gbrbm = gbrbm.initialize(nucleotide);
gbrbm = gbrbm.pretrain(nucleotide,[1e-8,1e-4],1e-2,1e6);

save('gbrbm_nucleotide.mat','gbrbm');
