% 2017-02-15
% author£ºZhangYong, 24452861@qq.com

clear all;
close all;
format compact;

screen_message = 'Preparing train data set ......'
[train_data] = import_mnist('mnist.mat');
screen_message = 'Preparing train data set ...... ok!'

screen_message = 'ZYM initialized ......'
zym = DML.ZYM([784,4096]);
screen_message = 'ZYM initialized ...... ok!'

screen_message = 'ZYM is training ......'
learn_rate_min = 1e-10;
learn_rate_max = 1e-1;
max_iteration = 1e6;
zym = zym.train(train_data,learn_rate_min,learn_rate_max,max_iteration);
screen_message = 'ZYM is training ...... ok!'

screen_message = 'ZYM is saving ......'
save('zym_trained.mat');
screen_message = 'ZYM is saving ...... ok!'
