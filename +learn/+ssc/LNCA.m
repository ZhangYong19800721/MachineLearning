classdef LNCA < learn.neural.PerceptionS
    %Linear NCA (Linear Neighbor Component Analysis)
    %  �����ڷ�������
    
    properties(Access=private)
        t; % �о�����
    end
    
    methods
        %% ���캯��
        function obj = LNCA(i,o) 
            obj@learn.neural.PerceptionS([i,o,i]); % ���ø���Ĺ��캯��
            obj.t = zeros(o,1); % ��ʼ���о�����
        end
    end
    
    methods
        %% ����
        function c = encode(obj,points,option)
            %% ����ѡ��
            if nargin <= 2
                option = 'real';
            end
            
            %% ��ʼ��
            [~,N] = size(points); % N��������
            
            %% �������
            code = obj.do(points,1); % ִ���������
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
            for d = 1:D
                center = learn.cluster.KMeansPlusPlus(code(d,:),2);
                obj.t(d) = sum(center)/2;
            end
            obj.t = reshape(obj.t,[],1);
        end
    end
    
end

