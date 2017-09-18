classdef NCA < learn.neural.PerceptionS
    %NCA NCA (Neighbor Component Analysis)
    %  邻分量分析
    
    properties(Access=private)
        thresh; % 量化判决门限
        clayer; % 指示编码层的位置
    end
    
    methods
        %% 构造函数
        function obj = R_NL_NCA(configure) 
            obj@learn.neural.PerceptionS(configure); % 调用父类的构造函数
            obj.clayer = (length(configure) - 1)/2;
        end
    end
    
    methods
        %% 编码
        function c = encode(obj,points,option)
            if nargin <= 2
                option = 'binary';
            end
            [~,N] = size(points); % N样本个数
            [~,c] = obj.do(points); % 执行正向计算
            if strcmp(option,'binary')
                c = c{obj.clayer} > repmat(obj.thresh,1,N);
            elseif strcmp(option,'real')
                c = c{obj.clayer};
            else
                assert(false);
            end
        end
        
        %% 计算判决门限
        function obj = findt(obj,points)
            code = obj.encode(points,'real');
            [D,~] = size(code); % 编码维度
            parfor d = 1:D
                [a(d),b(d)] = learn.cluster.KMeansPlusPlus(code(d,:),2);
            end
            obj.thresh = (a + b)/2;
            obj.thresh = reshape(obj.thresh,[],1);
        end
    end
    
end

