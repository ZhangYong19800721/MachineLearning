classdef LMBPS
    % LMBPS 配合训练PerceptionS感知器的训练类
    % 使用Levenberg Marquardt BP算法
    
    properties
        points;
        labels;
        percep; 
    end
    
    methods % 构造函数
        function obj = LMBPS(points,labels,percep)
            obj.points = points;
            obj.labels = labels;
            obj.percep = percep;
        end
    end
    
    methods
        function y = object(obj,x)
            obj.percep.weight = x;
            predict = obj.percep.do(obj.points);
            y = sum(sum((obj.labels - predict).^2));
        end
        
        function [H,G] = hessen(obj,x)
            %% 初始化
            obj.percep.weight = x; % 设置参数
            [K,~] = size(obj.labels); % K标签维度
            [~,N] = size(obj.points); % N样本个数
            P = numel(obj.percep.weight); % P参数个数
            M = length(obj.percep.num_hidden); % M层数
            H = zeros(P,P); % Hessian矩阵
            G = zeros(P,1); % 梯度向量
            s = cell(1,M); % 敏感性
            w = cell(1,M); cw = cell(1,M); % 权值
            b = cell(1,M); cb = cell(1,M); % 偏置
            for m = 1:M % 取得每一层的权值和偏置值
                [w{m},cw{m}] = obj.percep.getw(m);
                [b{m},cb{m}] = obj.percep.getb(m);
            end
            
            %% 计算Jacobi矩阵，Hessian矩阵，Gradient梯度
            [~,a] = obj.percep.do(obj.points); % 执行正向计算
            for n = 1:N
                %% 反向传播敏感性
                s{M} = diag(-a{M}(:,n).*(1-a{M}(:,n))); % 计算顶层的敏感性
                for m = (M-1):-1:1  % 反向传播
                    s{m} = s{m+1} * w{m+1} * diag(a{m}(:,n).*(1-a{m}(:,n)));
                end
                
                %% 计算Jacobi矩阵
                J = zeros(K,P);
                for m = 1:M
                    if m == 1
                        Jw = s{m} * kron(eye(obj.percep.num_hidden{m}),obj.points(:,n)'); % 敏感性对权值的导数
                        Jb = s{m}; 
                    else
                        Jw = s{m} * kron(eye(obj.percep.num_hidden{m}),a{m-1}(:,n)'); % 敏感性对权值的导数 
                        Jb = s{m};
                    end
                    J(:,cw{m}) = Jw;
                    J(:,cb{m}) = Jb;
                end
                
                %% 计算Hessian矩阵、Gradient梯度
                H = H + J'*J;
                G = G + 2*J'*(obj.labels(:,n) - a{M}(:,n));
            end
        end
    end
end

