classdef Weak_Line < learn.Weak
    %WEAK_LINE ������������
    % ���������������̳���learn.Weak�����Boost�㷨ʹ��
    
    properties
        w; % Ȩֵ
        b; % ƫ��
    end
    
    methods
        function c = predict(obj, points)
            c = obj.w * points + repmat(obj.b,1,size(points,2));
            c = c > 0;
        end
    end
    
end

