classdef Weak_Line < learn.Weak
    %WEAK_LINE ������������
    % ���������������̳���learn.Weak�����AdaBoostʹ��
    
    properties
        w; % Ȩֵ
        b; % ƫ��
    end
    
    methods
        function c = predict(obj, points)
            c = obj.w * points + repmat(obj.b,1,size(points,2));
            c(c > 0) = +1; c(c <= 0) = -1;
        end
    end
    
end

