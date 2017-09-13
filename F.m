classdef F
    %F 此处显示有关此类的摘要
    %   此处显示详细说明
    
    properties
    end
    
    methods
        function y = object(obj,x)
            y = 3 * x.^3 - 4 .* x + 2;
        end
    end
    
end

