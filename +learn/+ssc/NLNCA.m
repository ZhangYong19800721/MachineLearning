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
            point_a = obj.points(:,a); % 取第a个点
            point_b = obj.points(:,obj.simset{a}); % 取相似点集
            point_z = obj.points(:,obj.difset{a}); % 取不同点集
            
            %% 计算实数编码
            code_a = obj.encode(point_a);
            code_b = obj.encode(point_b);
            code_z = obj.encode(point_z);
            
            %% a点到其他所有点的距离
            dis_ab = sum((code_b - repmat(code_a,1,size(code_b,2))).^2,1); 
            dis_az = sum((code_z - repmat(code_a,1,size(code_z,2))).^2,1); 
            exp_ab = exp(-dis_ab); % a点到相似点距离的负指数函数
            exp_az = exp(-dis_az); % a点到不同点距离的负指数函数
            sum_ex = sum([exp_ab exp_az]);
            
            %% a点到相似点的概率
            p_ab = exp_ab / sum_ex;
            
            %% 计算目标函数
            y = obj.lamdax * sum(p_ab) + (1 - obj.lamdax) * cross_entropy;
        end
        
        %% 计算梯度
        function g = gradient(obj,x,a)
            %% 初始化
            obj.weight = x; % 首先设置参数
            point_a = obj.points(:,a); % 取第a个点
            point_b = obj.points(:,obj.simset{a}); % 取相似点集
            point_z = obj.points(:,obj.difset{a}); % 取不同点集
            
            %% 计算实数编码
            code_a = obj.encode(point_a);
            code_b = obj.encode(point_b);
            code_z = obj.encode(point_z);
            
            %% a点到其他所有点的距离
            dis_ab = sum((code_b - repmat(code_a,1,size(code_b,2))).^2,1); 
            dis_az = sum((code_z - repmat(code_a,1,size(code_z,2))).^2,1); 
            exp_ab = exp(-dis_ab); % a点到相似点距离的负指数函数
            exp_az = exp(-dis_az); % a点到不同点距离的负指数函数
            sum_ex = sum([exp_ab exp_az]);
            
            %% a点到其他所有点的概率
            p_ab = exp_ab / sum_ex; % a点到相似点的概率
            p_az = exp_az / sum_ex; % a点到不同点的概率
            p_a  = sum(p_ab); % a点到相似点的概率和
            
            %% 计算顶层的敏感性
            
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
    end
    
end

