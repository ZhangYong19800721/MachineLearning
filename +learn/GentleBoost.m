classdef GentleBoost
    %GENTLEBOOST GentleBoost算法
    %  参考“Robust Real-time Object Detection”
    
    properties
    end
    
    methods
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
            
            c = wc.predict(points); % 使用弱分类器对所有的数据点进行分类
            epsilon = sum(weight .* abs(c - labels)); % 计算在weight分布条件下的分类误差
            beda = epsilon / (1 - epsilon); 
            obj.weak{1+length(obj.weak)} = wc;
            obj.alfa = [obj.alfa log(1/beda)];
            w = weight .* (beda.^(1 - c~=labels)); % 更新权值
            w = w ./ sum(w);
        end
        
        function y = predict(obj,points)  % 使用经过训练的模型判断数据点的分类
            %PREDICT 对数据点进行分类，正例为1，反例为0
            %
            H = length(obj.alfa); [~,N] = size(points); % H弱分类器的个数，N数据点数
            c = logical(H,N); % 存储弱分类器的分类结果
            for h=1:H, c(h,:) = obj.weak{h}.predict(points); end
            y = obj.alfa * c >= sum(obj.alfa) / 2;
        end
    end
end

