classdef StumpSSC
    %StumpSSC 配合BoostSSC使用的弱分类器
    %   fm = a * [(x1(k) > t) == (x2(k) > t)] + b
    
    properties
        a; b; k; t;
    end
    
    methods(Access = private)
        function y = compare(obj,points1,points2)
            % compare 计算两个点之间的弱分类值
            c1 = points1(obj.k,:) > obj.t;
            c2 = points2(obj.k,:) > obj.t;
            y = obj.a * xor(c1,c2) + obj.b;
        end
        
        function y = compute(obj,points,paridx)
            % compute 计算两个点之间的弱分类值
            c = points(obj.k,:) > obj.t;
            y = obj.a * xor(c(paridx(1,:)),c(paridx(2,:))) + obj.b;
        end
    end
end

