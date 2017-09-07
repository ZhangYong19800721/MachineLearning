clear all
close all

load('nucleotides.mat');
[~,N] = size(nucleotide);

labels_neg = [];
for k = 1:10
    sequenc_idx = 1:N;
    shuffle_idx = randperm(N);
    labels_neg = [labels_neg [sequenc_idx; shuffle_idx; -ones(1,N)]];
end


load('labels_pos.mat');



