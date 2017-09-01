classdef Weak_LineR
    %Weak_LineR 线性实数弱分类器
    % 线性实数弱分类器
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
        end
        
        function f = compute(obj, points)
            b = obj.predict(points);
            f = zeros(size(b));
            f(b) = obj.x; f(~b) = obj.y;
        end
        
        function b = predict(obj, points)
            [~,N] = size(points);
            z = obj.w * points + repmat(obj.b,1,N);
            b = z > 0;
        end
    end
    
end

