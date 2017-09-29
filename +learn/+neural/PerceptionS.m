classdef PerceptionS
    %PERCEPTIONS 感知器
    %  最后一层输出为sigmoid神经元，采用交叉熵作为目标函数，可以使用CG、BFGS算法进行寻优
    
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
        function obj = PerceptionS(configure)
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
        %% 梯度计算
        function g = gradient(obj,x,m)
            %% 初始化
            [D,S,M] = size(obj.points); % D数据维度，S样本点数，M样本批数
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
                s = cell(1,L); % 敏感性
                w = cell(1,L); iw = cell(1,L); % 权值
                b = cell(1,L); ib = cell(1,L); % 偏置
                for l = 1:L % 取得每一层的权值和偏置值
                    [w{l},iw{l}] = obj.getw(l);
                    [b{l},ib{l}] = obj.getb(l);
                end
                
                %% 计算梯度
                [y,a] = obj.do(minibatch); % 执行正向计算
                s{L} = (y - minilabel)'; % 计算顶层的敏感性
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
                    g(ib{l},1) = g(ib{l},1) + sum(s{l},1)';
                end
            end
        end
        
        %% 计算目标函数
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
                z = obj.compute(minibatch);
                z(z<=0) = eps; z(z>=1) = 1 - eps;
                y = minilabel .* log(z) + (1-minilabel) .* log(1-z); % 计算交叉熵
                y = -sum(sum(y));
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
		
		function y = compute(obj,x,L)
            % 多层感知器的计算过程
            % y 是最后的输出
            
            if nargin <= 2
                L = length(obj.num_hidden); % 得到层数
            end
            
            for l = 1:L
                w = obj.getw(l);
                b = obj.getb(l);
                y = learn.tools.sigmoid(w * x + repmat(b,1,size(x,2)));
                x = y;
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
                a{l} = learn.tools.sigmoid(w * x + repmat(b,1,size(x,2)));
                x = a{l};
            end
            
            y = a{L};
        end
        
        function [w,r] = getw(obj,m)
            r = obj.star_w_idx{m}:obj.stop_w_idx{m};
            w = reshape(obj.weight(r),obj.num_hidden{m},obj.num_visual{m});
        end
        
        function [b,r] = getb(obj,m)
            r = obj.star_b_idx{m}:obj.stop_b_idx{m};
            b = reshape(obj.weight(r),[],1);
        end
        
        function obj = train(obj,points,labels,parameters)
            if nargin <= 3
                parameters = [];
                disp('调用train函数时没有给出参数集，将使用默认参数集');
            end
            
            if ~isfield(parameters,'algorithm')
                parameters.algorithm = 'CG';
                disp(sprintf('没有algorithm参数，将使用默认值%s',parameters.algorithm));
            end
            
            %% 绑定训练数据
            obj.points = points;
            obj.labels = labels;
            
            %% 寻优
            if strcmp(parameters.algorithm,'CG')
                obj.weight = learn.optimal.minimize_cg(obj,obj.weight,parameters);
            elseif strcmp(parameters.algorithm,'BFGS')
                obj.weight = learn.optimal.minimize_bfgs(obj,obj.weight,parameters);
            elseif strcmp(parameters.algorithm,'LM')
                obj.weight = learn.optimal.minimize_lm(obj,obj.weight,parameters);
			elseif strcmp(parameters.algorithm,'ADAM')
                obj.weight = learn.optimal.minimize_adam(obj,obj.weight,parameters);
			elseif strcmp(parameters.algorithm,'SGD')
                obj.weight = learn.optimal.minimize_sgd(obj,obj.weight,parameters);
			elseif strcmp(parameters.algorithm,'GD')
                obj.weight = learn.optimal.minimize_g(obj,obj.weight,parameters);
            end
            
            %% 解除绑定
            obj.points = [];
            obj.labels = [];
        end
    end
    
    methods(Static)
        function [] = unit_test1()
            clear all;
            close all;
            rng(1);
            
            N = 2000;
            x = linspace(-2,2,N);
            k = 2;
            f = @(x)0.5 + 0.5 * sin(k * pi * x / 4);
            l = f(x);
            
            configure = [1,12,1];
            p = learn.neural.PerceptionS(configure);
            p = p.initialize();
            
            figure(1);
            plot(x,l); hold on;
            plot(x,p.do(x)); hold off;
            
            paramters.algorithm = 'BFGS';
            % paramters.algorithm = 'CG';
            paramters.epsilon = 1e-3;
            paramters.max_it = 1e6;
            p = p.train(x,l,paramters);
            
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
            
            x = [0.8 0.2];
            l = [0.2 0.8];
            
            configure = [1,2,1];
            p = learn.neural.PerceptionS(configure);
            p = p.initialize();
            
            figure(1);
            plot(x,l); hold on;
            plot(x,p.do(x)); hold off;
            
            parameters.epsilon = 1e-8;
            p = p.train(x,l,parameters);
            
            figure(3);
            y = p.do(x);
            plot(x,l,'b'); hold on;
            plot(x,y,'r.'); hold off;
            
            disp(sprintf('error:%f',sum(sum((l - y).^2))));
        end
    end
end