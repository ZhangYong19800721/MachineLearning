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
        c = predict(obj, points) % ����ֵ����Ϊ1��0����ʾ�����ݵ����Ϊ��������
    end
end

