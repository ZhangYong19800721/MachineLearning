classdef QuadraticSSC
    %QUADRATICSSC 配合BoostProSSC使用的弱分类器
    %   实现二次函数型的弱分类器
    %   二次函数的形式为f(x) = x'*A*x+B*x+C
    %   当f(x)>0时判为正例，当f(x)<0时判为反例
    
    properties
        A; B; C; a; b;
    end
    
    methods(Access = public)
        function y = compare(obj,points1,points2)
            % compare 计算两个点之间的弱分类值
            % 弱分类器判为正例时返回a+b
            % 弱分类器判为反例时返回b
            
            v1 = learn.tools.quadratic(obj.A,obj.B,obj.C,points1);
            v2 = learn.tools.quadratic(obj.A,obj.B,obj.C,points2);
            c1 = v1 > 0;
            c2 = v2 > 0;
            y = obj.a * (c1==c2) + obj.b;
        end
        
        function y = compute(obj,points,paridx)
            % compute 计算两个点之间的弱分类值
            % 弱分类器判为正例时返回a+b
            % 弱分类器判为反例时返回b
            
            v = learn.tools.quadratic(obj.A,obj.B,obj.C,points);
            c = v > 0; c1 = c(paridx(1,:)); c2 = c(paridx(2,:));
            y = obj.a * (c1==c2) + obj.b;
        end
    end
end

