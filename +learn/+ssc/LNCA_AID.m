classdef LNCA_AID
    %LNCA_AID �����ڷ������������Ż�������
    %  ���㺯��ֵ���ݶ�
    
    properties(Access=private)
        points; % ���ݵ�
        labels; % ���Ʊ�ǩ
        simset; % ���Ƽ���
        lnca;   % ������
    end
    
    methods
        %% ���캯��
        function obj = LNCA_AID(points,labels,lnca) 
            obj.points = points;
            obj.labels = labels;
            obj.lnca = lnca;
            
            %% �������Ƽ���
            [~,N] = size(points);
            I = labels(1,:); J = labels(2,:);
            parfor n = 1:N
                index = (J == n | I == n);
                simset{n} = setdiff(union(I(index),J(index)),n);
            end
            obj.simset = simset;
        end
    end
    
    methods
        %% ����Ŀ�꺯��
        function y = object(obj,x)
            %% ��ʼ��
            [~,N] = size(obj.points);
            obj.rnlnca.weight = x; % �������ò���
            
            %% ����ʵ������
            code = obj.lnca.encode(obj.points); % ����ʵ������
            
            %% ����Ŀ�꺯��
            y = 0;
            for a = 1:N
                D = sum((code - repmat(code(:,a),1,N)).^2,1); % ����a�㵽����������ľ���
                E = exp(-D); S = sum(E) - 1; % ���㸺ָ������
                b = obj.simset{a};
                y = y + sum(E(b) / S);
            end
        end
        
        %% �����ݶ�
        function g = gradient(obj,x)
            %% ��ʼ��
            [~,N] = size(obj.points); % N��������
            W = size(obj.lnca.weight,1); % W��������
            obj.rnlnca.weight = x; % �������ò���
            g = zeros(W,1); % ��ʼ���ݶ�
            
            %% ����ʵ������
            code = obj.rnlnca.encode(obj.points); 
            
            %% �����ݶ�
            for a = 1:N
                
            end
        end
    end
    
    methods(Static)
        %% ��Ԫ����
        function [] = unit_test() 
            
        end
    end
end

