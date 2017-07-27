function [center_points, labels]=KMeans(points,K)
% K,�������
% points,���������㣨ѵ�����ݣ�
% center_points, ����ѵ����õ������ĵ�
% points_labels, ���е�ѵ�����ݼӱ�ǩ��Ľ��

    [D, N]=size(points);  %D������ά����N�����������

    % �����ʼ��K�����ĵ�
    center_points = zeros(D,K);  %�����ʼ�������յ�����ÿһ�������λ��
    for d=1:D
        max_value = max(points(d,:));    %ÿһά������
        min_value = min(points(d,:));    %ÿһά��С����
        center_points(d,:) = min_value+(max_value-min_value)*rand(1,K);  %�����ʼ��
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

