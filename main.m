clear all;
close all;
format compact;

SURF = ConstructSurfData('D:\imagebase\','*.jpeg');
VV = ConstructVisualVocabulary(2048,SURF,5);

% data1 = rand(2,1e4) + repmat([2.5;-0.5],1,1e4);
% data2 = rand(2,1e4) + repmat([-3.5;-0.5],1,1e4);
% 
% data = [data1 data2];
% 
% [c,l] = Cluster.KMeansPlusPlus(data,2);
% 
% plot(data1(1,:),data1(2,:),'y+');
% hold;
% plot(data2(1,:),data2(2,:),'g+');
% plot(c(1,:),c(2,:),'Mo');
% grid on;
