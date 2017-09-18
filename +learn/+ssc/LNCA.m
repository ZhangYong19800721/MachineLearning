classdef LNCA < learn.neural.PerceptionS
    %Linear NCA (Linear Neighbor Component Analysis)
    %  �����ڷ�������
    
    properties(Access=private)
        t; % �о�����
    end
    
    methods
        %% ���캯��
        function obj = LNCA(o,i) 
            obj@learn.neural.PerceptionS([i,o,i]); % ���ø���Ĺ��캯��
            obj.t = zeros(o,1); % ��ʼ���о�����
        end
    end
    
    methods
        %% ����
        function c = encode(obj,points,option)
            if nargin <= 2
                option = 'real';
            end
            [~,N] = size(points); % N��������
            code = obj.do(points,1);
            if strcmp(option,'binary')
                c = code > repmat(obj.t,1,N);
            elseif strcmp(option,'real')
                c = code;
            else
                assert(false);
            end
        end
        
        %% �����о�����
        function obj = findt(obj,points)
            code = obj.encode(points);
            [D,~] = size(code); % ����ά��
            parfor d = 1:D
                [a(d),b(d)] = learn.cluster.KMeansPlusPlus(code(d,:),2);
            end
            obj.t = (a + b)/2;
            obj.t = reshape(obj.t,[],1);
        end
    end
    
end

