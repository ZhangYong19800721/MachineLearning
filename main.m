clear all;
close all;

load('nucleotides.mat');
[D,N] = size(nucleotide);
S = 100; M = floor(N / S); N = S*M;
nucleotide = nucleotide(:,1:N);
nucleotide(nucleotide>0) = 1;
nucleotide = reshape(nucleotide,D,S,M);

configure = [D,1024,256];
sae = learn.SAE(configure);

parameters.learn_rate = [1e-6,1e-2];
parameters.weight_cost = 1e-4;
parameters.max_it = 1e6;
sae = sae.pretrain(nucleotide,parameters);
            
nucleotide = reshape(nucleotide,D,[]);
recon_nucleotide = sae.rebuild(nucleotide,'sample');
recon_nucleotide = recon_nucleotide > 0.5;

error = sum(sum((nucleotide - recon_nucleotide).^2)) / N;

save('sae_pretrained_DAN.mat','sae');
