classdef Weak
    %Weak ��������
    %   ��AdaBoost�����ʹ��
    
    properties
    end
    
    methods
        function obj = Weak()
        end
    end
    
    methods (Abstract)
        c = predict(obj, points) % ����ֵ����Ϊ+1��-1����ʾ�����ݵ����Ϊ��������
    end
end

