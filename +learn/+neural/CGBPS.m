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
        
        %% 梯度计算
        function g = gradient(obj,x)
            %% 初始化
            obj.percep.weight = x; % 初始化权值
            N = size(obj.points,2); % N样本个数
            P = size(obj.percep.weight,1); % P参数个数
            M = length(obj.percep.num_hidden); % M层数
            g = zeros(P,1); % 梯度
            s = cell(1,M); % 敏感性
            w = cell(1,M); % 权值
            b = cell(1,M); % 偏置
            for m = 1:M % 取得每一层的权值和偏置值
                w{m} = obj.percep.getw(m);
                b{m} = obj.percep.getb(m);
            end
            
            %% 计算梯度
            [~,a] = obj.percep.do(obj.points); % 执行正向计算
            s{M} = -2 * (obj.labels - a{M})' .* (a{M} .* (1 - a{M}))'; % 计算顶层的敏感性
            for m = (M-1):-1:1  % 反向传播敏感性
                sx = s{m+1}; wx = w{m+1}; ax = a{m}.*(1-a{m}); 
                sm = zeros(N,obj.percep.num_hidden{m});
                parfor n = 1:N
                    sm(n,:) = sx(n,:) * wx * diag(ax(:,n));
                end
                s{m} = sm;
            end
            
            for m = 1:M
                [~,cw] = obj.percep.getw(m);
                [~,cb] = obj.percep.getb(m);

                H = obj.percep.num_hidden{m};
                V = obj.percep.num_visual{m};
                
                sx = s{m}'; 
                if m == 1
                    px = obj.points';
                    gx = zeros(size(w{m}));
                    parfor n = 1:N
                        gx = gx + repmat(sx(:,n),1,V) .* repmat(px(n,:),H,1);
                    end
                else
                    ax = a{m-1}';
                    gx = zeros(size(w{m}));
                    parfor n = 1:N
                        gx = gx + repmat(sx(:,n),1,V) .* repmat(ax(n,:),H,1);
                    end
                end
                
                g(cw,1) = g(cw,1) + gx(:);
                g(cb,1) = g(cb,1) + sum(sx,2);
            end
            
            g = g ./ N;
        end
    end
    
end

