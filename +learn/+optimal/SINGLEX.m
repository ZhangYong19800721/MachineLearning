classdef SINGLEX
    %SINGLEX SINGLEX��һ������������F��������Ϊһ������������
    %
    
    properties
        F;
        x0;
        d0;
    end
    
    methods
        function obj = SINGLEX(F,x0,d0)
            obj.F = F; obj.x0 = x0; obj.d0 = d0;
        end
        
        function y = object(obj,x)
            y = obj.F.object(obj.x0 + x * obj.d0);
        end
    end
    
end

