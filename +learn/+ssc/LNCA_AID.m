classdef LNCA_AID
    %LNCA_AID 线性邻分量分析的最优化辅助类
    %  计算函数值和梯度
    
    properties(Access=private)
        points; % 数据点
        labels; % 相似标签
        simset; % 相似集合
        lnca;   % 神经网络
    end
    
    methods
        %% 构造函数
        function obj = LNCA_AID(points,labels,lnca) 
            obj.points = points;
            obj.labels = labels;
            obj.lnca = lnca;
            
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
            obj.rnlnca.weight = x; % 首先设置参数
            
            %% 计算实数编码
            code = obj.lnca.encode(obj.points); % 计算实数编码
            
            %% 计算目标函数
            y = 0;
            for a = 1:N
                D = sum((code - repmat(code(:,a),1,N)).^2,1); % 计算a点到所有其它点的距离
                E = exp(-D); E(a) = 0; S = sum(E); % 计算负指数函数
                b = obj.simset{a};
                y = y + sum(E(b) / S);
            end
        end
        
        %% 计算梯度
        function g = gradient(obj,x)
            %% 初始化
            [D,N] = size(obj.points); % N样本个数
            P = size(obj.lnca.weight,1); % P参数个数
            obj.rnlnca.weight = x; % 首先设置参数
            g = zeros(P,1); % 初始化梯度
            [A,r] = obj.lnca.getw(1); % 线性变换矩阵
            gA = zeros(size(A(:))); 
            
            %% 计算实数编码
            code = obj.rnlnca.encode(obj.points); 
            
            %% 计算梯度
            xpoints = obj.points; 
            xsimset = obj.simset;
            parfor a = 1:N
                d_az = sum((code - repmat(code(:,a),1,N)).^2,1); % 计算a点到所有其它点的距离
                e_az = exp(-d_az); e_az(a) = 0; s_az = sum(e_az); % 计算负指数函数
                p_az = e_az / s_az; % 计算a点到所有其它点的概率
                x_az = repmat(sqrt(p_az),D,1) .* (repmat(xpoints(:,a),1,N) - xpoints); % Xaz
                gA = gA + sum(p_az(xsimset{a})) * (x_az*x_az') - x_az(:,xsimset{a})*x_az(:,xsimset{a})';
            end
            
            gA = 2*A*gA;
            g(r) = gA(:);
        end
    end
    
    methods(Static)
        %% 单元测试
        function [] = unit_test() 
            
        end
    end
end

