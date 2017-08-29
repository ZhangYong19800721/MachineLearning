classdef Weak_LineR < learn.Weak
    %Weak_LineR ����ʵ����������
    % ����ʵ�������������̳���learn.Weak��
    % ���RealAdaBoost�㷨ʹ��
    % 
    
    properties
        w; % Ȩֵ
        b; % ƫ��
        x; % �������������ݵ���Ϊ����ʱ�ķ���ֵ
        y; % �������������ݵ���Ϊ����ʱ�ķ���ֵ
    end
    
    methods
        function obj = Weak_LineR()
            obj.x = true;
            obj.y = false;
        end
        
        function c = predict(obj, points)
            z = obj.w * points + repmat(obj.b,1,size(points,2));
            positive = z > 0;
            c = obj.y * ones(size(z)); c(positive) = obj.x; 
        end
    end
    
end

