classdef DiscreteAdaBoost1
    %DiscreteAdaBoost 实现Discrete AdaBoost算法
    %  参考论文"Improved Boosting Algorithms Using Confidence-rated Predictions"
    
    properties
        weak; % 包含若干个弱分类器的cell数组
        alfa; % 每个弱分类器的投票权值
    end
    
    methods
        function obj = DiscreteAdaBoost1()
        end
    end
    
    methods
        function [obj,w]  = ensemble(obj,points,labels,weight,wc)
            %ensemble 组合弱分类器，找到一个最优的弱分类器，然后通过此函数组合起来
            %   输入：
            %   points 表示数据点
            %   labels 标签+1或-1
            %   weight 概率分布权值
            %   wc 新加入的弱分类器
            % 
            %   输出：
            %   obj 训练后的adaboost对象
            %   w 更新后的权值
            
            c = wc.predict(points); k(c) = 1; k(~c) = -1;
            r = weight * (labels .* k)';
            beda = 0.5 * log((1 + r)/(1 - r));
            obj.weak{1+length(obj.weak)} = wc;
            obj.alfa = [obj.alfa beda];
            w = weight .* exp(-beda * labels .* k);
            w = w ./ sum(w);
        end
        
        function y = predict(obj,points)  % 使用经过训练的模型判断数据点的分类
            %PREDICT 判断两个点是否相似，相似为+1，不相似为-1
            %
            H = length(obj.alfa); [~,N] = size(points); % H弱分类器的个数，N数据点数
            c = logical(H,N); % 存储弱分类器的分类结果
            for h=1:H,c(h,:) = obj.weak{h}.predict(points);end
            k(c) = 1; k(~c) = -1;
            y = obj.alfa * k > 0;
        end
    end
end

