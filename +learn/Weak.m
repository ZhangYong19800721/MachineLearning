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
        c = predict(obj, points) % 返回值必须为1或0，表示将数据点分类为正例或反例
    end
end

