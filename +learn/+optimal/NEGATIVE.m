classdef NEGATIVE
    %NEGATIVE NEGATIVE��һ��������������F�����Ŀ�꺯�����ݶȵļ���������Fȡ��
    %
    
    properties
        F;
    end
    
    methods
        function obj = NEGATIVE(F)
            obj.F = F;
        end
        
        function y = object(obj,x)
            y = -obj.F.object(x);
        end
        
        function g = gradient(obj,x)
            g = -obj.F.gradient(x);
        end
    end
    
end

