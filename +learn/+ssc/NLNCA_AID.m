classdef NLNCA_AID
    %NCA_AID 正则非线性邻分量分析的最优化辅助类
    %  计算神经网络的函数值和梯度
    
    properties(Access=private)
        points; % 数据点
        labels; % 相似标签
        simset; % 相似集合
        difset; % 不同集合
        rnlnca; % 神经网络
    end
    
    methods
        %% 构造函数
        function obj = NLNCA_AID(points,labels,rnlnca) 
            obj.points = points;
            obj.labels = labels;
            obj.rnlnca = rnlnca;
            
            labels_pos = labels(:,labels(3,:)==+1);
            labels_neg = labels(:,labels(3,:)==-1);
            
            %% 计算相似集合
            [~,N] = size(points);
            I = labels_pos(1,:); J = labels_pos(2,:);
            parfor n = 1:N
                index = (J == n | I == n);
                simset{n} = setdiff(union(I(index),J(index)),n);
            end
            obj.simset = simset;
            
            %% 计算不同集合
            I = labels_neg(1,:); J = labels_neg(2,:);
            parfor n = 1:N
                index = (J == n | I == n);
                difset{n} = setdiff(union(I(index),J(index)),n);
            end
            obj.difset = difset;
        end
    end
    
    methods
        %% 计算目标函数
        function y = object(obj,x)
            %% 初始化
            [~,N] = size(obj.points);
            obj.rnlnca.weight = x; % 首先设置参数
            
            %% 计算实数编码
            code = obj.rnlnca.encode(obj.points); % 计算实数编码
            
            %% 计算目标函数
            y = 0; 
            for a = 1:N
                d1 = sum((code(:,obj.simset{a}) - repmat(code(:,a),1,numel(obj.simset{a}))).^2,1);
                d2 = sum((code(:,obj.difset{a}) - repmat(code(:,a),1,numel(obj.difset{a}))).^2,1);
                e1 = exp(-d1); se1 = sum(e1);
                e2 = exp(-d2); se2 = sum(e2);
                y = y + se1 / (se1 + se2);
            end
            y = y / N;
        end
        
        %% 计算梯度
        function g = gradient(obj,x)
            %% 初始化
            [~,N] = size(obj.points); % N样本个数
            P = size(obj.rnlnca.weight,1); % P参数个数
            obj.rnlnca.weight = x; % 首先设置参数
            g = zeros(P,1); % 初始化梯度
            
            %% 计算实数编码
            code = obj.rnlnca.encode(obj.points); 
            
            %% 计算梯度
            for a = 1:N
                
            end
        end
    end
    
    methods(Static)
        %% 单元测试
        function [] = unit_test() 
            
        end
    end
end

