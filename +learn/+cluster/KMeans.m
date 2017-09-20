function [center_points, labels]=KMeans(points,K)
% K,分类个数
% points,数据样本点（训练数据）
% center_points, 经过训练后得到的中心点
% points_labels, 所有的训练数据加标签后的结果

    [D, N]=size(points);  %D是数据维数，N是样本点个数

    % 随机初始化K个中心点
    center_points = zeros(D,K);  %随机初始化，最终迭代到每一类的中心位置
    for d=1:D
        max_value = max(points(d,:));    %每一维最大的数
        min_value = min(points(d,:));    %每一维最小的数
        center_points(d,:) = min_value+(max_value-min_value)*rand(1,K);  %随机初始化
    end
    
    distance_matrix = zeros(K,N);
    
    while true
        for k = 1:K
            center_point_k = repmat(center_points(:,k),1,N);
            delta = points - center_point_k;
            distance_matrix(k,:) = sqrt(sum(delta.^2));
        end
        
        [~,min_idx] = min(distance_matrix);
        
        old_center_points = center_points;
        for k = 1:K
            center_points(:,k) = sum(points(:,min_idx == k),2) / sum(min_idx == k);
        end
        
        if norm(old_center_points - center_points) <= 0.001
            for k = 1:K
                center_point_k = repmat(center_points(:,k),1,N);
                delta = points - center_point_k;
                distance_matrix(k,:) = sqrt(sum(delta.^2));
            end
            
            [~,labels] = min(distance_matrix);
            break;
        end
    end
end

