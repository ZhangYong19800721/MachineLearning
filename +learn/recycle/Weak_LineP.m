classdef Weak_LineP < learn.Weak
    %Weak_LineP ���Ը�����������
    % ���������������̳���learn.Weak�����RealAdaBoost�㷨ʹ��
    
    properties
        w; % Ȩֵ
        b; % ƫ��
    end
    
    methods
        function c = predict(obj, points)
            c = obj.w * points + repmat(obj.b,1,size(points,2));
            c = c ./ norm(obj.w,2);
            c = 2 * learn.sigmoid(c) - 1;
        end
    end
    
end

