classdef Weak_LineR
    %Weak_LineR ����ʵ����������
    % ����ʵ����������
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
        end
        
        function f = compute(obj, points)
            b = obj.predict(points);
            f = zeros(size(b));
            f(b) = obj.x; f(~b) = obj.y;
        end
        
        function b = predict(obj, points)
            [~,N] = size(points);
            z = obj.w * points + repmat(obj.b,1,N);
            b = z > 0;
        end
    end
    
end

