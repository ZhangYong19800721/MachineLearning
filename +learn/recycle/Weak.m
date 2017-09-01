classdef Weak
    %Weak 弱分类器
    %   与AdaBoost类配合使用
    
    properties
    end
    
    methods
        function obj = Weak()
        end
    end
    
    methods (Abstract)
        % 对于离散adaboost返回值必须为1或0，表示将数据点分类为正例或反例
        % 对于实数adaboost返回值介于[0,1]之间，表示数据点为正例的概率
        c = predict(obj, points) 
    end
end

