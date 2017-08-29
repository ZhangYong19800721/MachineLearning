classdef GentleBoost
    %GENTLEBOOST GentleBoost算法
    %  参考论文"Additive Logistic Regression a Statistical View of Boosting"
    %  这篇论文非常重要，揭示了boost算法和加性统计模型在本质上是相同的，分析
    %  了AdaBoost、RealBoost、LogitBoost、GentleBoost算法。
    
    properties
        weak; % 包含若干个弱分类器的cell数组
    end
    
    methods(Access = private)
        function [obj,w]  = ensemble(obj,points,labels,weight,wc)
            %ensemble 组合弱分类器
            %   输入：
            %   points 表示数据点
            %   labels 标签1或0
            %   weight 概率分布权值
            %   wc 新加入的弱分类器
            % 
            %   输出：
            %   obj 训练后的adaboost对象
            %   w 更新后的权值
            
            f = wc.predict(points); % 首先计算弱分类器的输出
            obj.weak{1+length(obj.weak)} = wc; % 将弱分类器加入到boost组中
            w = weight .* (-labels .* f); % 更新权值
            w = w ./ sum(w); % 归一化权值
        end
    end
    
    methods(Access = public)
        function y = compute(obj,points)
            M = length(obj.weak); [~,N] = size(points); % M弱分类器的个数，N数据点数
            f = logical(M,N); % 存储弱分类器的分类结果
            for m=1:M, f(m,:) = obj.weak{m}.predict(points); end
            y = sum(f);
        end
        
        function y = predict(obj,points)  % 使用经过训练的模型判断数据点的分类
            %PREDICT 对数据点进行分类，正例为1，反例为0
            %
            y = obj.compute(points) > 0;
        end
        
        function obj = train(obj,points,labels,M)
            % train 训练GentleBoost模型
            % 输入：
            % points 数据点
            % labels 标签，+1或-1
            % M 弱分类器的个数
            % 输出：
            % obj 经过训练的boost对象
            
            [~,N] = size(points); % 得到数据点数目
            weight = ones(1,N) / N; % 初始化权值
            
            
        end
    end
end

