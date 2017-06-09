function VV = ConstructVisualVocabulary(K,SURF)
%CONSTRUCTVISUALVOCABULARY �����Ӿ��ʻ��
%   K, �Ӿ��ʻ��Ĵ�С
%   SURF, SURF�������ݼ���ÿһ����һ��64ά��SURF����������
%   VV��Visual Vocabulary���Ӿ��ʻ��ÿһ�б�ʾһ��64ά�Ӿ��ʻ㣬��K��

    object_value = inf;
    for n = 1:10
        [center, label] = Cluster.KMeansPlusPlus(SURF,K);
        
        sum_dis = zeros(1,K);
        for k = 1:K
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

