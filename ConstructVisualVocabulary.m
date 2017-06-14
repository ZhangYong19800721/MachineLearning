function VV = ConstructVisualVocabulary(K,SURF,Attempt)
%CONSTRUCTVISUALVOCABULARY 构建视觉词汇表
%   K, 视觉词汇表的大小
%   SURF, SURF特征数据集，每一列是一个64维的SURF特征向量。
%   Attempt, Kmeans++执行的次数
%   VV，Visual Vocabulary，视觉词汇表，每一列表示一个64维视觉词汇，共K列
    message = 'constructing visual vocabulary ...'

    object_value = inf;
    for n = 1:Attempt
        attempt = n
        [center, label] = Cluster.KMeansPlusPlus(SURF,K);
        
        sum_dis = zeros(1,K);
        parfor k = 1:K
            idx_k = (label == k);
            delta = repmat(center(:,k),1,sum(idx_k)) - SURF(:,idx_k);
            sum_dis(k) = sum(sqrt(sum(delta.^2)));
        end
        new_object_value = sum(sum_dis);
        if new_object_value < object_value
            object_value = new_object_value;
            VV = center;
        end
    end
end

