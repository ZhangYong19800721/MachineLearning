clear all;
close all;

load('nucleotides.mat');
load('labels_pos.mat');
load('labels_neg.mat');
labels = [labels labels_neg];
clear labels_neg;

ssc = learn.ssc.GentleAdaBoostSSCHam();
ssc = ssc.train(nucleotide,labels,0.95,128);

save('ssc.mat','ssc');



