classdef NLNCA < learn.neural.PerceptionS
    %NLNCA Non-Linear NCA (Neighbor Component Analysis)
    %  �������ڷ�������
    
    properties(Access=private)
        t; % �����о�����
        c; % ָʾ������λ��
    end
    
    methods
        %% ���캯��
        function obj = NLNCA(configure) 
            obj@learn.neural.PerceptionS(configure); % ���ø���Ĺ��캯��
            obj.c = (length(configure) - 1)/2;
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
            code = obj.do(points,obj.c); % ִ���������
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
            code = obj.encode(points); % �������
            [D,~] = size(code); % ����ά��
            parfor d = 1:D
                [a(d),b(d)] = learn.cluster.KMeansPlusPlus(code(d,:),2);
            end
            obj.t = (a + b)/2;
            obj.t = reshape(obj.t,[],1);
        end
    end
    
end

