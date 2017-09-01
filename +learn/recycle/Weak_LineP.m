classdef Weak_LineP < learn.Weak
    %Weak_LineP 线性概率弱分类器
    % 线性弱分类器，继承自learn.Weak。配合RealAdaBoost算法使用
    
    properties
        w; % 权值
        b; % 偏置
    end
    
    methods
        function c = predict(obj, points)
            c = obj.w * points + repmat(obj.b,1,size(points,2));
            c = c ./ norm(obj.w,2);
            c = 2 * learn.sigmoid(c) - 1;
        end
    end
    
end

