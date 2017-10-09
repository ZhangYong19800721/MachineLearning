classdef NLNCA < learn.neural.PerceptionS
    %NLNCA Non-Linear NCA (Neighbor Component Analysis)
    %  非线性邻分量分析
    
    properties(Access=private)
        thresh; % 量化判决门限
        co_idx; % 指示编码层的位置
        simset; % 相似集合
        difset; % 不同集合
        lamdax; % 正则因子
    end
    
    methods
        %% 构造函数
        function obj = NLNCA(configure,k,lamda) 
            obj@learn.neural.PerceptionS(configure); % 调用父类的构造函数
            obj.co_idx = k;
            obj.lamdax = lamda;
        end
    end
    
    methods
        %% 计算目标函数
        function y = object(obj,x,a)
            %% 初始化
            obj.weight = x; % 首先设置参数
            N = size(obj.points,2); 
            a = 1+mod(a,N);
            point_a = obj.points(:,a); Na = 1; % 取第a个点
            point_b = obj.points(:,obj.simset{a}); Nb = numel(obj.simset{a}); % 取相似点集
            point_z = obj.points(:,obj.difset{a}); Nz = numel(obj.difset{a}); % 取不同点集
            
            %% 计算实数编码
            [y_a,f_a] = obj.do(point_a); f_a = f_a{obj.co_idx};
            [y_b,f_b] = obj.do(point_b); f_b = f_b{obj.co_idx};
            [y_z,f_z] = obj.do(point_z); f_z = f_z{obj.co_idx};
            f_ab = repmat(f_a,1,size(f_b,2)) - f_b;
            f_az = repmat(f_a,1,size(f_z,2)) - f_z;
            
            %% a点到其他所有点的距离
            dis_ab = sum(f_ab.^2,1); 
            dis_az = sum(f_az.^2,1); 
            exp_ab = exp(-dis_ab); % a点到相似点距离的负指数函数
            exp_az = exp(-dis_az); % a点到不同点距离的负指数函数
            sum_ex = sum([exp_ab exp_az]);
            
            %% a点到相似点的概率
            p_ab = exp_ab / sum_ex;
            p_a = sum(p_ab); % a点到相似点的概率和
            
            %% 计算交叉熵
            y_a(y_a<=0) = eps; y_a(y_a>=1) = 1 - eps;
            y_b(y_b<=0) = eps; y_b(y_b>=1) = 1 - eps;
            y_z(y_z<=0) = eps; y_z(y_z>=1) = 1 - eps;
            e_a = point_a .* log(y_a) + (1-point_a) .* log(1-y_a);
            e_b = point_b .* log(y_b) + (1-point_b) .* log(1-y_b);
            e_z = point_z .* log(y_z) + (1-point_z) .* log(1-y_z);
            c_e = -sum(sum([e_a e_b e_z])) / (Na+Nb+Nz);
            
            %% 计算目标函数
            % y = obj.lamdax * p_a + (1 - obj.lamdax) * c_e;
            y = p_a;
        end
        
        %% 计算梯度
        function g = gradient(obj,x,a)
            %% 初始化
            obj.weight = x; % 首先设置参数
            L = length(obj.num_hidden); % L层数
            g = zeros(size(obj.weight,1),1); % 梯度
            point_a = obj.points(:,a); % 取第a个点
            point_b = obj.points(:,obj.simset{a}); % 取相似点集
            point_z = obj.points(:,obj.difset{a}); % 取不同点集
            s = cell(1,L); % 敏感性
            w = cell(1,L); iw = cell(1,L); % 权值
            b = cell(1,L); ib = cell(1,L); % 偏置
            for l = 1:L % 取得每一层的权值和偏置值
                [w{l},iw{l}] = obj.getw(l);
                [b{l},ib{l}] = obj.getb(l);
            end
            
            %% 计算实数编码
            [~,f_a] = obj.do(point_a); c_a = f_a{obj.co_idx};
            [~,f_b] = obj.do(point_b); c_b = f_b{obj.co_idx};
            [~,f_z] = obj.do(point_z); c_z = f_z{obj.co_idx};
            c_ab = repmat(c_a,1,size(c_b,2)) - c_b;
            c_az = repmat(c_a,1,size(c_z,2)) - c_z;
            
            %% a点到其他所有点的距离
            dis_ab = sum(c_ab.^2,1); 
            dis_az = sum(c_az.^2,1); 
            exp_ab = exp(-dis_ab); % a点到相似点距离的负指数函数
            exp_az = exp(-dis_az); % a点到不同点距离的负指数函数
            sum_ex = sum([exp_ab exp_az]);
            
            %% a点到其他所有点的概率
            p_ab = exp_ab / sum_ex; % a点到相似点的概率
            p_az = exp_az / sum_ex; % a点到不同点的概率
            p_a  = sum(p_ab); % a点到相似点的概率和
            
            %% 计算顶层的敏感性
            s_a = c_a .* (1-c_a);
            s_b = c_b .* (1-c_b);
            s_z = c_z .* (1-c_z);
            s_ab = repmat(s_a,1,size(s_b,2)) - s_b;
            s_az = repmat(s_a,1,size(s_z,2)) - s_z;
            s1 = sum(repmat(p_az,size(c_az,1),1) .* (2 * c_az .* s_az),2);
            s2 = sum(repmat(p_ab,size(c_ab,1),1) .* (2 * c_ab .* s_ab),2);
            s{obj.co_idx} = p_a * (s1+s2) - s2;
            
            %% 反向传播敏感性
            for l = L:-1:1  
                if l < obj.co_idx
                    s{l} = (s{l+1} * w{l+1}) .* (a{l}.*(1-a{l}))';
                elseif l > obj.co_idx
                    H = obj.num_hidden{l};
                    s{l} = zeros(1,H);
                end
            end
            
            for l = 1:L
                H = obj.num_hidden{l};
                V = obj.num_visual{l};
                if l == 1
                    minibatch = [point_a point_b point_z];
                    gx = reshape(repmat(s{l}',V,1),H,V,S) .* reshape(repelem(minibatch,H,1),H,V,S);
                elseif l <= obj.co_idx
                    minibatch = [f_a{l-1} f_b{l-1} f_z{l-1}];
                    gx = reshape(repmat(s{l}',V,1),H,V,S) .* reshape(repelem(minibatch,H,1),H,V,S);     
                else
                    gx = zeros(H,V);
                end
                gx = sum(gx,3);
                g(iw{l},1) = g(iw{l},1) + gx(:);
                g(ib{l},1) = g(ib{l},1) + sum(s{l},1)';
            end
        end
        
        %% 编码
        function c = encode(obj,points,option)
            %% 参数选项
            if nargin <= 2
                option = 'real'; 
            end
            
            %% 初始化
            [~,N] = size(points); % N样本个数
            
            %% 计算编码
            code = obj.compute(points,obj.co_idx); % 执行正向计算
            if strcmp(option,'binary')
                c = code > repmat(obj.thresh,1,N);
            elseif strcmp(option,'real')
                c = code;
            else
                assert(false);
            end
        end
        
        %% 计算判决门限
        function obj = findt(obj,points)
            code = obj.encode(points); % 计算编码
            [D,~] = size(code); % 编码维度
            for d = 1:D
                center = learn.cluster.KMeansPlusPlus(code(d,:),2);
                obj.t(d) = sum(center)/2;
            end
            obj.t = reshape(obj.t,[],1);
        end
        
        %% 训练
        function obj = train(obj,points,labels,parameters)
            %% 参数检查与设置
            if nargin <= 3
                parameters = [];
                disp('调用train函数时没有给出参数集，将使用默认参数集');
            end
            
            %% 绑定训练数据
            obj.points = points;
            obj.labels = labels;
            
            %% 组织训练数据
            labels_pos = labels(1:2,labels(3,:)==+1); % 相似标签
            labels_neg = labels(1:2,labels(3,:)==-1); % 不同标签
            
            %% 计算相似集合
            [~,N] = size(points);
            I = labels_pos(1,:); J = labels_pos(2,:);
            parfor n = 1:N
                idx = (J == n | I == n);
                simset{n} = setdiff(union(I(idx),J(idx)),n);
            end
            obj.simset = simset;
            
            %% 计算不同集合
            I = labels_neg(1,:); J = labels_neg(2,:);
            parfor n = 1:N
                idx = (J == n | I == n);
                difset{n} = setdiff(union(I(idx),J(idx)),n);
            end
            obj.difset = difset;
            
            %% 寻优
            obj.weight = learn.optimal.maximize_sadam(obj,obj.weight,parameters);
            
            %% 解除绑定
            obj.points = [];
            obj.labels = [];
        end
    end
    
    methods(Static)
        function [] = unit_test()
            clear all;
            close all;
            rng(1);
            
            load('images.mat'); points = points(1:(32*32),:); points = double(points) / 255;
            load('labels_pos.mat');
            load('labels_neg.mat'); 
            labels = [labels_pos labels_neg];
            
            configure = [1024,500,64,500,1024];
            nca = learn.ssc.NLNCA(configure,2,0.99);
            nca = nca.initialize();
            
            nca = nca.train(points,labels);
        end
    end
end

