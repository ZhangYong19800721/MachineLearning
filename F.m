classdef F
    %F �˴���ʾ�йش����ժҪ
    %   �˴���ʾ��ϸ˵��
    
    properties
    end
    
    methods
        function g = gradient(obj,x,i)
            g = 2 * (x - [3 4]');
        end
        
        function y = object(obj,x,i)
            y = sum((x - [3 4]').^2);
        end
    end
    
end

