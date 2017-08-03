clear all;
close all;

load('nucleotides.mat');

[D,N] = size(nucleotide);
shuffle_idx = randperm(N);
nucleotide = nucleotide(:,shuffle_idx);

nucleotide = nucleotide ./ max(max(nucleotide));

minibatch_size = 100;
minibatch_num  = floor(N / minibatch_size);

for minibatch_idx = 1:minibatch_num
    star_col = (minibatch_idx-1)*minibatch_size + 1;
    stop_col = (minibatch_idx-1)*minibatch_size + minibatch_size;
    points{minibatch_idx} = nucleotide(:,star_col:stop_col);
end

clear nucleotide;

% configure = [8688,1024,64];
rbm = learn.RestrictedBoltzmannMachine(D,1024);
rbm = rbm.initialize(points,-100,-4);
rbm = rbm.pretrain(points,1e-5,1e-1,1e6);
