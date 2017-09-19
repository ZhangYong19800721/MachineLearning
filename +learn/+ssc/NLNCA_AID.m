classdef NLNCA_AID
    %NCA_AID ����������ڷ������������Ż�������
    %  ����������ĺ���ֵ���ݶ�
    
    properties(Access=private)
        points; % ���ݵ�
        labels; % ���Ʊ�ǩ
        simset; % ���Ƽ���
        difset; % ��ͬ����
        rnlnca; % ������
    end
    
    methods
        %% ���캯��
        function obj = NLNCA_AID(points,labels,rnlnca) 
            obj.points = points;
            obj.labels = labels;
            obj.rnlnca = rnlnca;
            
            labels_pos = labels(:,labels(3,:)==+1);
            labels_neg = labels(:,labels(3,:)==-1);
            
            %% �������Ƽ���
            [~,N] = size(points);
            I = labels_pos(1,:); J = labels_pos(2,:);
            parfor n = 1:N
                index = (J == n | I == n);
                simset{n} = setdiff(union(I(index),J(index)),n);
            end
            obj.simset = simset;
            
            %% ���㲻ͬ����
            I = labels_neg(1,:); J = labels_neg(2,:);
            parfor n = 1:N
                index = (J == n | I == n);
                difset{n} = setdiff(union(I(index),J(index)),n);
            end
            obj.difset = difset;
        end
    end
    
    methods
        %% ����Ŀ�꺯��
        function y = object(obj,x)
            %% ��ʼ��
            [~,N] = size(obj.points);
            obj.rnlnca.weight = x; % �������ò���
            
            %% ����ʵ������
            code = obj.rnlnca.encode(obj.points); % ����ʵ������
            
            %% ����Ŀ�꺯��
            y = 0; 
            for a = 1:N
                d1 = sum((code(:,obj.simset{a}) - repmat(code(:,a),1,numel(obj.simset{a}))).^2,1);
                d2 = sum((code(:,obj.difset{a}) - repmat(code(:,a),1,numel(obj.difset{a}))).^2,1);
                e1 = exp(-d1); se1 = sum(e1);
                e2 = exp(-d2); se2 = sum(e2);
                y = y + se1 / (se1 + se2);
            end
            y = y / N;
        end
        
        %% �����ݶ�
        function g = gradient(obj,x)
            %% ��ʼ��
            [~,N] = size(obj.points); % N��������
            P = size(obj.rnlnca.weight,1); % P��������
            obj.rnlnca.weight = x; % �������ò���
            g = zeros(P,1); % ��ʼ���ݶ�
            
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

