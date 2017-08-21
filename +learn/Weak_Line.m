classdef Weak_Line < learn.Weak
    %WEAK_LINE 线性弱分类器
    % 线性弱分类器，继承自learn.Weak。配合Boost算法使用
    
    properties
        w; % 权值
        b; % 偏置
    end
    
    methods
        function c = predict(obj, points)
            c = obj.w * points + repmat(obj.b,1,size(points,2));
            c = c > 0;
        end
    end
    
end

