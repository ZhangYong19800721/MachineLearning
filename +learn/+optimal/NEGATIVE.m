classdef NEGATIVE
    %NEGATIVE NEGATIVE是一个包裹器，包裹F，其对目标函数和梯度的计算结果等于F取反
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
        
        function v = vector(obj,x)
            v = -obj.F.vector(x);
        end
        
        function j = jacobi(obj,x)
            j = -obj.F.jacobi(x);
        end
    end
    
end

