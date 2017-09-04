classdef StumpSSC
    %StumpSSC 配合BoostSSC使用的弱分类器
    %   fm = a * [(x1(k) > t) == (x2(k) > t)] + b
    
    properties
        a; b; k; t;
    end
    
    methods(Access = public)
        function y = compare(obj,points1,points2)
            % compare 计算两个点之间的弱分类值
            % 弱分类器判为正例时返回a+b
            % 弱分类器判为反例时返回b
            c1 = points1(obj.k,:) > obj.t;
            c2 = points2(obj.k,:) > obj.t;
            y = obj.a * (c1==c2) + obj.b;
        end
        
        function y = compute(obj,points,paridx)
            % compute 计算两个点之间的弱分类值
            % 弱分类器判为正例时返回a+b
            % 弱分类器判为反例时返回b
            c = points(obj.k,:) > obj.t;
            c1 = c(paridx(1,:)); c2 = c(paridx(2,:));
            y = obj.a * (c1==c2) + obj.b;
        end
    end
end

