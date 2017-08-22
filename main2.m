clear all;
close all;

N = 1e4;
[points,labels] = learn.GenerateData.type1(N);

figure;
group1 = points(:,labels== 1);
group2 = points(:,labels==-1);
plot(group1(1,:),group1(2,:),'+'); hold on;
plot(group2(1,:),group2(2,:),'*'); hold off;




