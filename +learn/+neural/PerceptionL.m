classdef PerceptionL
    %PERCEPTIONL 感知器
    % 最后一层输出为线性神经元
    
    properties
        weight;      % 一维数组，所有层的权值和偏置值都包含在这个一维数组中[P,1]
        num_hidden;  % num_hidden{m}表示第m层的隐神经元个数
        num_visual;  % num_visual{m}表示第m层的显神经元个数
        star_w_idx;  % star_w_idx{m}表示第m层的权值的起始位置
        stop_w_idx;  % stop_w_idx{m}表示第m层的权值的结束位置
        star_b_idx;  % star_b_idx{m}表示第m层的偏置的起始位置
        stop_b_idx;  % stop_b_idx{m}表示第m层的偏置的结束位置
        points; % 训练数据
        labels; % 训练标签
    end
    
    methods % 构造函数
        function obj = PerceptionL(configure)
            % Perception 构造函数
            M = length(configure) - 1;
            obj.num_hidden{1} = configure(2);
            obj.num_visual{1} = configure(1);
            obj.star_w_idx{1} = 1;
            obj.stop_w_idx{1} = obj.num_hidden{1} * obj.num_visual{1};
            obj.star_b_idx{1} = obj.stop_w_idx{1} + 1;
            obj.stop_b_idx{1} = obj.stop_w_idx{1} + obj.num_hidden{1};
            
            for m = 2:M
                obj.num_hidden{m} = configure(m+1);
                obj.num_visual{m} = configure(m+0);
                obj.star_w_idx{m} = obj.stop_b_idx{m-1} + 1;
                obj.stop_w_idx{m} = obj.stop_b_idx{m-1} + obj.num_hidden{m} * obj.num_visual{m};
                obj.star_b_idx{m} = obj.stop_w_idx{m} + 1;
                obj.stop_b_idx{m} = obj.stop_w_idx{m} + obj.num_hidden{m};
            end
        end
    end
    
    methods
        %% 计算梯度
        function g = gradient(obj,x,m)
            %% 初始化
            [D,S,M] = size(obj.points);
            P = size(obj.weight,1); % P参数个数
            L = length(obj.num_hidden); % L层数
            g = zeros(P,1); % 梯度
            
            if nargin <= 2 % 没有给出i参数，默认对全部训练数据求导数
                for m = 1:M
                    g = g + obj.gradient(x,m);
                end
            else
                m = 1 + mod(m,M);
                minibatch = obj.points(:,:,m); % 取一个minibatch
                minilabel = obj.labels(:,:,m); % 取一个minibatch
                obj.weight = x; % 初始化权值
                s  = cell(1,L); % 敏感性
                w  = cell(1,L); % 权值
                iw = cell(1,L); % 权值索引
                b  = cell(1,L); % 偏置
                ib = cell(1,L); % 偏置索引
                for l = 1:L     % 取得每一层的权值和偏置值
                    [w{l},iw{l}] = obj.getw(l);
                    [b{l},ib{l}] = obj.getb(l);
                end
                
                %% 计算梯度
                [~,a] = obj.do(minibatch); % 执行正向计算
                s{l} = -2 * (minilabel - a{l})'; % 计算顶层的敏感性
                for l = (L-1):-1:1  % 反向传播敏感性
                    s{l} = (s{l+1} * w{l+1}) .* (a{l}.*(1-a{l}))';
                end
                
                for l = 1:L
                    H = obj.num_hidden{l};
                    V = obj.num_visual{l};
                    if l == 1
                        gx = reshape(repmat(s{l}',V,1),H,V,S) .* reshape(repelem(minibatch,H,1),H,V,S);
                    else
                        gx = reshape(repmat(s{l}',V,1),H,V,S) .* reshape(repelem(a{l-1}   ,H,1),H,V,S);
                    end
                    gx = sum(gx,3);
                    g(iw{l},1) = g(iw{l},1) + gx(:);
                    g(ib{l},1) = g(ib{l},1) + sum(s{l})';
                end
            end
        end
        
        %% 计算函数值
        function y = object(obj,x,m)
            %% 初始化
            [D,S,M] = size(obj.points);
            
            if nargin <= 2 % 没有给出m参数，默认对全部训练数据求导数
                y = 0;
                for m = 1:M
                    y = y + obj.object(x,m);
                end
            else
                obj.weight = x; % 初始化权值
                m = 1 + mod(m,M);
                minibatch = obj.points(:,:,m); % 取一个minibatch
                minilabel = obj.labels(:,:,m); % 取一个minibatch
                
                %% 计算目标函数
                z = obj.do(minibatch);
                y = sum(sum((minilabel - z).^2));
            end
        end
        
        %% 计算hessen矩阵
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
                s{M} = -eye(K); % 计算顶层的敏感性
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
        
        function obj = initialize(obj)
            M = length(obj.num_hidden); % 得到层数
            for m = 1:M
                obj.weight(obj.star_w_idx{m}:obj.stop_w_idx{m},1) = ...
                    0.01 * randn(size([obj.star_w_idx{m}:obj.stop_w_idx{m}]')); % 将权值初始化为0附近的随机数
                obj.weight(obj.star_b_idx{m}:obj.stop_b_idx{m},1) = ...
                    zeros(size([obj.star_b_idx{m}:obj.stop_b_idx{m}]')); % 将偏置值初始化为0
            end
        end
        
        function [y,a] = do(obj,x,L)
            % 多层感知器的计算过程
            % y 是最后的输出
            
            if nargin <= 2
                L = length(obj.num_hidden); % 得到层数
            end
           
            a = cell(1,L);          
            for l = 1:L
                w = obj.getw(l);
                b = obj.getb(l);
                n = w * x + repmat(b,1,size(x,2));
                if l < L
                    a{l} = learn.tools.sigmoid(n);
                    x = a{l};
                else
                    a{l} = n;
                    y = n;
                end
            end
        end
        
        function [w,r] = getw(obj,m)
            r = obj.star_w_idx{m}:obj.stop_w_idx{m};
            w = reshape(obj.weight(r),obj.num_hidden{m},obj.num_visual{m});
        end
        
        function [b,r] = getb(obj,m)
            r = obj.star_b_idx{m}:obj.stop_b_idx{m};
            b = reshape(obj.weight(r),[],1);
        end
        
        function obj = train(obj,points,labels)
            %% 绑定训练数据
            obj.points = points;
            obj.labels = labels;
            
            %% 寻优
            obj.weight = learn.optimal.minimize_cg(obj,obj.weight);
            
            %% 解除绑定
            obj.points = points;
            obj.labels = labels;
        end
    end
    
    methods(Static)
        function p = unit_test1()
            clear all;
            close all;
            rng(1);
            
            N = 2000;
            x = linspace(-2,2,N);
            k = 4;
            f = @(x)sin(k * pi * x / 4);
            l = f(x);
            
            configure = [1,6,1];
            p = learn.neural.PerceptionL(configure);
            p = p.initialize();
            
            figure(1);
            plot(x,l); hold on;
            plot(x,p.do(x)); hold off;
            
            lmbp = learn.neural.LMBPL(x,l,p);
            weight = learn.optimal.minimize_lm(lmbp,p.weight);
            p.weight = weight;
            
            figure(3);
            y = p.do(x);
            plot(x,l,'b'); hold on;
            plot(x,y,'r.'); hold off;
            
            disp(sprintf('error:%f',sum(sum((l - y).^2))));
        end
        
        function [] = unit_test2()
            clear all;
            close all;
            rng(1);
            
            N = 2000;
            x = linspace(-2,2,N);
            k = 4;
            f = @(x)sin(k * pi * x / 4);
            l = f(x);
            
            configure = [1,6,1];
            p = learn.neural.PerceptionL(configure);
            p = p.initialize();
            
            figure(1);
            plot(x,l); hold on;
            plot(x,p.do(x)); hold off;
            
            p = p.train(reshape(x,1,N,1),reshape(l,1,N,1));
            
            figure(3);
            y = p.do(x);
            plot(x,l,'b'); hold on;
            plot(x,y,'r.'); hold off;
            
            disp(sprintf('error:%f',sum(sum((l - y).^2))));
        end
        
        function p = unit_test3()
            clear all;
            close all;
            rng(1);
            
            N = 2000;
            x = linspace(-2,2,N);
            k = 4;
            f = @(x)sin(k * pi * x / 4);
            l = f(x);
            
            configure = [1,6,1];
            p = learn.neural.PerceptionL(configure);
            p = p.initialize();
            
            figure(1);
            plot(x,l); hold on;
            plot(x,p.do(x)); hold off;
            
            cgbp = learn.neural.CGBPL(x,l,p);
            weight = learn.optimal.minimize_bfgs(cgbp,p.weight);
            p.weight = weight;
            
            figure(3);
            y = p.do(x);
            plot(x,l,'b'); hold on;
            plot(x,y,'r.'); hold off;
            
            disp(sprintf('error:%f',sum(sum((l - y).^2))));
        end
        
        function [] = unit_test4()
            clear all;
            close all;
            rng(1);
            
            N = 2000;
            x = linspace(-2,2,N);
            k = 4;
            f = @(x)sin(k * pi * x / 4);
            l = f(x);
            
            configure = [1,6,1];
            p = learn.neural.PerceptionL(configure);
            p = p.initialize();
            
            figure(1);
            plot(x,l); hold on;
            plot(x,p.do(x)); hold off;
            
            
            p.points = reshape(x,1,20,100);
            p.labels = reshape(l,1,20,100);
            weight = learn.optimal.minimize_adam(p,p.weight);
            p.weight = weight;
            
            figure(3);
            y = p.do(x);
            plot(x,l,'b'); hold on;
            plot(x,y,'r.'); hold off;
            
            disp(sprintf('error:%f',sum(sum((l - y).^2))));
        end
    end
end