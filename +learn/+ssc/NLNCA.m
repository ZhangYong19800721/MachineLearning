classdef NLNCA < learn.neural.PerceptionS
    %NLNCA Non-Linear NCA (Neighbor Component Analysis)
    %  非线性邻分量分析
    
    properties(Access=private)
        t; % 量化判决门限
        c; % 指示编码层的位置
    end
    
    methods
        %% 构造函数
        function obj = NLNCA(configure) 
            obj@learn.neural.PerceptionS(configure); % 调用父类的构造函数
            obj.c = (length(configure) - 1)/2;
        end
    end
    
    methods
        %% 编码
        function c = encode(obj,points,option)
            %% 参数选项
            if nargin <= 2
                option = 'real'; 
            end
            
            %% 初始化
            [~,N] = size(points); % N样本个数
            
            %% 计算编码
            code = obj.do(points,obj.c); % 执行正向计算
            if strcmp(option,'binary')
                c = code > repmat(obj.t,1,N);
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
            parfor d = 1:D
                [a(d),b(d)] = learn.cluster.KMeansPlusPlus(code(d,:),2);
            end
            obj.t = (a + b)/2;
            obj.t = reshape(obj.t,[],1);
        end
    end
    
end

