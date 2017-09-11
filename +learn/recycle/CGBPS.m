classdef CGBPS
    % Conjugate Gradient BP Sigmoid 感知器的训练类
    % 使用Conjugate Gradient BP算法训练(共轭梯度反向传播算法)
    
    properties
        points;
        labels;
        percep; 
    end
    
    methods % 构造函数
        function obj = CGBPS(points,labels,percep)
            obj.points = points;
            obj.labels = labels;
            obj.percep = percep;
        end
    end
    
    methods
        function y = object(obj,x)
            % 计算目标函数
            obj.percep.weight = x;
            predict = obj.percep.do(obj.points);
            y = sum(sum((obj.labels - predict).^2));
        end
        
        function g = gradient(obj,x)
            %% 初始化
            obj.percep.weight = x; % 初始化权值
            N = size(obj.points,2); % N样本个数
            P = size(obj.percep.weight,1); % P参数个数
            M = length(obj.percep.num_hidden); % M层数
            g = zeros(P,N); % 每一个数据样本的梯度
            
            %% 计算梯度
            [~,a] = obj.percep.do(obj.points); % 执行正向计算
            s{M} = -2 * (obj.labels - a{M})' .* (a{M} .* (1 - a{M}))'; % 计算顶层的敏感性
            for m = (M-1):-1:1  % 反向传播敏感性
                weight = obj.percep.getw(m+1);
                s{m} = s{m+1} * weight * diag(a{m}.*(1-a{m}));
            end
            
            parfor n = 1:N
                s = cell(1,M); % s用来记录敏感性
                s{M} = -2 * (obj.labels(:,n) - a{M})' .* (a{M} .* (1 - a{M}))'; % 计算顶层的敏感性
                for m = (M-1):-1:1  % 反向传播敏感性
                    weight = obj.percep.getw(m+1);
                    s{m} = s{m+1} * weight * diag(a{m}.*(1-a{m}));
                end
                
                for m = 1:M
                    [~,cw] = obj.percep.getw(m);
                    [~,cb] = obj.percep.getb(m);
                    
                    H = obj.percep.num_hidden{m};
                    V = obj.percep.num_visual{m};
                    
                    if m == 1
                        f2w = repmat(s{m}',1,V) .* repmat(obj.points(:,n)',H,1);
                    else
                        f2w = repmat(s{m}',1,V) .* repmat(a{m-1}',H,1);
                    end
                    
                    g(cw,1) = g(cw,1) + reshape(f2w,[],1);
                    g(cb,1) = g(cb,1) + s{m}';
                end
            end
            
            g = g ./ N;
        end
    end
    
end

