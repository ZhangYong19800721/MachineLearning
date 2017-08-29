classdef Weak_LineR < learn.Weak
    %Weak_LineR 线性实数弱分类器
    % 线性实数弱分类器，继承自learn.Weak。
    % 配合RealAdaBoost算法使用
    % 
    
    properties
        w; % 权值
        b; % 偏置
        x; % 弱分类器将数据点判为正例时的返回值
        y; % 弱分类器将数据点判为反例时的返回值
    end
    
    methods
        function obj = Weak_LineR()
            obj.x = true;
            obj.y = false;
        end
        
        function c = predict(obj, points)
            z = obj.w * points + repmat(obj.b,1,size(points,2));
            positive = z > 0;
            c = obj.y * ones(size(z)); c(positive) = obj.x; 
        end
    end
    
end

