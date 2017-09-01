clear all;
close all;

import_similarity_labels;

load('nucleotides.mat');

K = 1;
subplot(2,1,1); bar(nucleotide(:,labels(1,K))); subplot(2,1,2); bar(nucleotide(:,labels(2,K)));

