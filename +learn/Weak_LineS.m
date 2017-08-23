classdef Weak_LineS < learn.Weak
    %Weak_LineS ���Ը�����������
    % ���������������̳���learn.Weak�����RealAdaBoost�㷨ʹ��
    
    properties
        w; % Ȩֵ
        b; % ƫ��
    end
    
    methods
        function c = predict(obj, points)
            c = obj.w * points + repmat(obj.b,1,size(points,2));
            c = c ./ norm(obj.w,2);
            c = learn.sigmoid(c);
            c = 0.5 * log(c./(1-c));
        end
    end
    
end

