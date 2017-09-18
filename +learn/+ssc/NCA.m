classdef NCA < learn.neural.PerceptionS
    %NCA NCA (Neighbor Component Analysis)
    %  �ڷ�������
    
    properties(Access=private)
        thresh; % �����о�����
        clayer; % ָʾ������λ��
    end
    
    methods
        %% ���캯��
        function obj = R_NL_NCA(configure) 
            obj@learn.neural.PerceptionS(configure); % ���ø���Ĺ��캯��
            obj.clayer = (length(configure) - 1)/2;
        end
    end
    
    methods
        %% ����
        function c = encode(obj,points,option)
            if nargin <= 2
                option = 'binary';
            end
            [~,N] = size(points); % N��������
            [~,c] = obj.do(points); % ִ���������
            if strcmp(option,'binary')
                c = c{obj.clayer} > repmat(obj.thresh,1,N);
            elseif strcmp(option,'real')
                c = c{obj.clayer};
            else
                assert(false);
            end
        end
        
        %% �����о�����
        function obj = findt(obj,points)
            code = obj.encode(points,'real');
            [D,~] = size(code); % ����ά��
            parfor d = 1:D
                [a(d),b(d)] = learn.cluster.KMeansPlusPlus(code(d,:),2);
            end
            obj.thresh = (a + b)/2;
            obj.thresh = reshape(obj.thresh,[],1);
        end
    end
    
end

