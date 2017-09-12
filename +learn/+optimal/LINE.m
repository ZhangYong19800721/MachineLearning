classdef LINE
    %LINE LINE是一个包裹器，包裹F，将F函数包裹为一个线搜索函数
    %
    
    properties
        F;
        x0;
        d0;
    end
    
    methods
        function obj = LINE(F,x0,d0)
            obj.F = F; obj.x0 = x0; obj.d0 = d0;
        end
        
        function y = object(obj,x)
            y = obj.F.object(obj.x0 + x * obj.d0);
        end
    end
    
end

