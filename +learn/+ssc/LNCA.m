classdef LNCA < learn.neural.PerceptionS
    %Linear NCA (Linear Neighbor Component Analysis)
    %  线性邻分量分析
    
    properties(Access=private)
        t; % 判决门限
    end
    
    methods
        %% 构造函数
        function obj = LNCA(i,o) 
            obj@learn.neural.PerceptionS([i,o,i]); % 调用父类的构造函数
            obj.t = zeros(o,1); % 初始化判决门限
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
            code = obj.do(points,1); % 执行正向计算
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
            code = obj.encode(points);
            [D,~] = size(code); % 编码维度
            for d = 1:D
                center = learn.cluster.KMeansPlusPlus(code(d,:),2);
                obj.t(d) = sum(center)/2;
            end
            obj.t = reshape(obj.t,[],1);
        end
    end
    
end

