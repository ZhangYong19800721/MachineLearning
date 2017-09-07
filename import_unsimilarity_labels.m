clear all
close all

load('nucleotides.mat');
[~,N] = size(nucleotide);

 %% 随机组合
labels_neg = [];
for k = 1:15
    sequenc_idx = 1:N;
    shuffle_idx = randperm(N);
    labels_neg = [labels_neg [sequenc_idx; shuffle_idx; -ones(1,N)]];
end

 %% 去掉两个序号相同的项
idx = labels_neg(1,:) == labels_neg(2,:);
labels_neg = labels_neg(:,~idx);

 %% 从不相似集合中去除已知的相似数据对
load('labels_pos.mat');
labels_pos = labels; % 载入相似标签数据对

[~,U] = size(labels_neg);
[~,S] = size(labels_pos);

select = false(1,U);
for u = 1:U
    disp(sprintf('process:%f',u/U));
    i = labels_neg(1,u);
    j = labels_neg(2,u);
    
    idx1 = labels_pos(1,   :) == i | labels_pos(2,   :) == i;
    idx2 = labels_pos(1,idx1) == j | labels_pos(2,idx1) == j;
    if sum(idx2) == 0
        select(u) = true;
    end
end

labels_neg = labels_neg(:,select);
save('labels_neg.mat','labels_neg');





