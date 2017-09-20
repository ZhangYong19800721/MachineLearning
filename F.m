classdef F
    %F 此处显示有关此类的摘要
    %   此处显示详细说明
    
    properties
    end
    
    methods
        function g = gradient(obj,x,i)
            g = 2 * (x - [3 4]');
        end
        
        function y = object(obj,x,i)
            y = sum((x - [3 4]').^2);
        end
    end
    
end

