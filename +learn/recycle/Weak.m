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
        % ������ɢadaboost����ֵ����Ϊ1��0����ʾ�����ݵ����Ϊ��������
        % ����ʵ��adaboost����ֵ����[0,1]֮�䣬��ʾ���ݵ�Ϊ�����ĸ���
        c = predict(obj, points) 
    end
end

