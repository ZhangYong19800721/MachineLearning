classdef NCA_AID
    %NCA_AID 正则非线性邻分量分析的最优化辅助类
    %  计算神经网络的函数值和梯度
    
    properties(Access=private)
        points; % 数据点
        labels; % 相似标签
        simset; % 相似集合
        rnlnca; % 神经网络
    end
    
    methods
        %% 构造函数
        function obj = R_NL_NCA_AID(points,labels,rnlnca) 
            obj.points = points;
            obj.labels = labels;
            obj.rnlnca = rnlnca;
            
            %% 计算相似集合
            [~,N] = size(points);
            I = labels(1,:); J = labels(2,:);
            parfor n = 1:N
                index = (J == n | I == n);
                simset{n} = setdiff(union(I(index),J(index)),n);
            end
            obj.simset = simset;
        end
    end
    
    methods
        %% 计算目标函数
        function y = object(obj,x)
            %% 初始化
            [~,N] = size(obj.points);
            
            %% 首先设置参数
            obj.rnlnca.weight = x;
            
            %% 计算实数编码
            code = obj.rnlnca.encode(obj.points,'real');
            y = 0;
            for a = 1:N
                dist = sum((code - repmat(code(:,a),1,N)).^2); % a点到其它所有点之间的距离
                prob = exp(-dist);
                for b = obj.simset{a}
                    p = prob(b) / (sum(prob) - 1);
                    y = y + p;
                end
            end
        end
        
        %% 计算梯度
        function g = gradient(obj,x)
            %% 初始化
            [~,N] = size(obj.points);
            
            %% 首先设置参数
            obj.rnlnca.weight = x;
            
            
        end
    end
    
    methods(Static)
        %% 单元测试
        function [] = unit_test() 
            disp('unit-test of learn.ssc.R_NL_NCA_AID');
        end
    end
end

