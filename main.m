clear all;
close all;

load('nucleotides.mat');

[D,N] = size(nucleotide);
shuffle_idx = randperm(N);
nucleotide = nucleotide(:,shuffle_idx);

S = 100; M = floor(N / S); N = S*M;
nucleotide = nucleotide(:,1:(S*M));

for minibatch_idx = 1:M
    star_col = (minibatch_idx-1)*minibatch_size + 1;
    stop_col = (minibatch_idx-1)*minibatch_size + minibatch_size;
    points{minibatch_idx} = nucleotide(:,star_col:stop_col);
end

clear nucleotide;

% configure = [8688,1024,64];
rbm = learn.RestrictedBoltzmannMachine(D,1024);
rbm = rbm.initialize(points,-100,-4);
rbm = rbm.pretrain(points,1e-5,1e-1,1e6);
