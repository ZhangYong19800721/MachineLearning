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
        
        function y = object(obj,x,m)
            y = -obj.F.object(x,m);
        end
        
        function g = gradient(obj,x,m)
            g = -obj.F.gradient(x,m);
        end
        
        function v = vector(obj,x,m)
            v = -obj.F.vector(x,m);
        end
        
        function j = jacobi(obj,x,m)
            j = -obj.F.jacobi(x,m);
        end
    end
    
end

