classdef LNCA_AID
    %LNCA_AID 线性邻分量分析的最优化辅助类
    %  计算函数值和梯度
    
    properties(Access=private)
        points; % 数据点
        labels; % 相似标签
        simset; % 相似集合
        difset; % 不同集合
        lnca;   % 神经网络
    end
    
    methods
        %% 构造函数
        function obj = LNCA_AID(points,labels,lnca) 
            obj.points = points;
            obj.labels = labels;
            obj.lnca = lnca;
            
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
            obj.lnca.weight = x; % 首先设置参数
            
            %% 计算实数编码
            code = obj.lnca.encode(obj.points); % 计算实数编码
            
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
            [D,N] = size(obj.points); % N样本个数
            P = size(obj.lnca.weight,1); % P参数个数
            obj.lnca.weight = x; % 首先设置参数
            g = zeros(P,1); % 初始化梯度
            [A,r] = obj.lnca.getw(1); % 线性变换矩阵
            
            %% 计算实数编码
            code = obj.lnca.encode(obj.points); 
            
            %% 计算梯度
            q = 0;
            for a = 1:N
                d1 = sum((code(:,obj.simset{a}) - repmat(code(:,a),1,numel(obj.simset{a}))).^2,1); % 计算a点到相似点的距离
                d2 = sum((code(:,obj.difset{a}) - repmat(code(:,a),1,numel(obj.difset{a}))).^2,1);
                e1 = exp(-d1); se1 = sum(e1); % 计算负指数函数
                e2 = exp(-d2); se2 = sum(e2);
                p1 = e1 / (se1 + se2); % 计算a点到相似点的概率
                p2 = e2 / (se1 + se2); 
                x_a1 = repmat(obj.points(:,a),1,numel(obj.simset{a})) - obj.points(:,obj.simset{a});
                x_a2 = repmat(obj.points(:,a),1,numel(obj.difset{a})) - obj.points(:,obj.difset{a});
                q1 = x_a1 * diag(p1) * x_a1';
                q2 = x_a2 * diag(p2) * x_a2';
                q = q + (se1 / (se1 + se2)) * (q1 + q2) - q1;
            end
            
            q = 2 * A * q / N;
            g(r) = q(:);
        end
    end
    
    methods(Static)
        %% 单元测试
        function [] = unit_test() 
            
        end
    end
end

