classdef SINGLEX
    %SINGLEX SINGLEX是一个包裹器，将F函数包裹为一个单变量函数
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

