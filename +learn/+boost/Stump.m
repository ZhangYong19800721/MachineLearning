classdef Stump
    %STUMP 配合GentleAdaBoost使用的弱分类器
    %   fm = a * (x(k) > t) + b
    
    properties
        a; b; k; t;
    end
    
    methods
        function y = compute(obj,points)
            y = obj.a * (points(obj.k,:) > obj.t) + obj.b;
        end
    end
    
end

