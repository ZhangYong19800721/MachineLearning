classdef NLNCA < learn.neural.PerceptionS
    %NLNCA Non-Linear NCA (Neighbor Component Analysis)
    %  �������ڷ�������
    
    properties(Access=private)
        thresh; % �����о�����
        co_idx; % ָʾ������λ��
        simset; % ���Ƽ���
        difset; % ��ͬ����
        lamdax; % ��������
    end
    
    methods
        %% ���캯��
        function obj = NLNCA(configure,k,lamda) 
            obj@learn.neural.PerceptionS(configure); % ���ø���Ĺ��캯��
            obj.co_idx = k;
            obj.lamdax = lamda;
        end
    end
    
    methods
        %% ����Ŀ�꺯��
        function y = object(obj,x,a)
            %% ��ʼ��
            obj.weight = x; % �������ò���
            N = size(obj.points,2); 
            a = 1+mod(a,N);
            point_a = obj.points(:,a); % ȡ��a����
            point_b = obj.points(:,obj.simset{a}); % ȡ���Ƶ㼯
            point_z = obj.points(:,obj.difset{a}); % ȡ��ͬ�㼯
            
            %% ����ʵ������
            [y_a,f_a] = obj.do(point_a); f_a = f_a{obj.co_idx};
            [y_b,f_b] = obj.do(point_b); f_b = f_b{obj.co_idx};
            [y_z,f_z] = obj.do(point_z); f_z = f_z{obj.co_idx};
            f_ab = repmat(f_a,1,size(f_b,2)) - f_b;
            f_az = repmat(f_a,1,size(f_z,2)) - f_z; 
            
            %% a�㵽�������е�ľ���
            dis_ab = sum(f_ab.^2,1); 
            dis_az = sum(f_az.^2,1); 
            exp_ab = exp(-dis_ab); % a�㵽���Ƶ����ĸ�ָ������
            exp_az = exp(-dis_az); % a�㵽��ͬ�����ĸ�ָ������
            sum_ex = sum([exp_ab exp_az]);
            
            %% a�㵽���Ƶ�ĸ���
            p_ab = exp_ab / sum_ex;
            p_a = sum(p_ab); % a�㵽���Ƶ�ĸ��ʺ�
            
            %% ���㽻����
            y_a(y_a<=0) = eps; y_a(y_a>=1) = 1 - eps;
            y_b(y_b<=0) = eps; y_b(y_b>=1) = 1 - eps;
            y_z(y_z<=0) = eps; y_z(y_z>=1) = 1 - eps;
            e_a = point_a * log(y_a) + (1-point_a) * log(1-y_a);
            e_b = point_b * log(y_b) + (1-point_b) * log(1-y_b);
            e_z = point_z * log(y_z) + (1-point_z) * log(1-y_z);
            e = -e_a-e_b-e_z;
            
            %% ����Ŀ�꺯��
            y = obj.lamdax * p_a + (1 - obj.lamdax) * e;
        end
        
        %% �����ݶ�
        function g = gradient(obj,x,a)
            %% ��ʼ��
            obj.weight = x; % �������ò���
            point_a = obj.points(:,a); % ȡ��a����
            point_b = obj.points(:,obj.simset{a}); % ȡ���Ƶ㼯
            point_z = obj.points(:,obj.difset{a}); % ȡ��ͬ�㼯
            
            %% ����ʵ������
            f_a = obj.compute(point_a,obj.co_idx);
            f_b = obj.compute(point_b,obj.co_idx);
            f_z = obj.compute(point_z,obj.co_idx);
            f_ab = repmat(f_a,1,size(f_b,2)) - f_b;
            f_az = repmat(f_a,1,size(f_z,2)) - f_z;
            
            %% a�㵽�������е�ľ���
            dis_ab = sum(f_ab.^2,1); 
            dis_az = sum(f_az.^2,1); 
            exp_ab = exp(-dis_ab); % a�㵽���Ƶ����ĸ�ָ������
            exp_az = exp(-dis_az); % a�㵽��ͬ�����ĸ�ָ������
            sum_ex = sum([exp_ab exp_az]);
            
            %% a�㵽�������е�ĸ���
            p_ab = exp_ab / sum_ex; % a�㵽���Ƶ�ĸ���
            p_az = exp_az / sum_ex; % a�㵽��ͬ��ĸ���
            p_a  = sum(p_ab); % a�㵽���Ƶ�ĸ��ʺ�
            
            %% ���㶥���������
            s_a = f_a.*(1-f_a);
            s_b = f_b.*(1-f_b);
            s_z = f_z.*(1-f_z);
            
            s_ab = repmat(s_a,1,size(s_b,2)) - s_b;
            s_az = repmat(s_a,1,size(s_z,2)) - s_z;
            
            s1 = sum(repmat(p_az,size(f_az,1),1) * 2 * f_az .* s_az,2);
            s2 = sum(repmat(p_ab,size(f_ab,1),1) * 2 * f_ab .* s_ab,2);
            
            s = p_a * (s1+s2) - s2;
        end
        
        %% ����
        function c = encode(obj,points,option)
            %% ����ѡ��
            if nargin <= 2
                option = 'real'; 
            end
            
            %% ��ʼ��
            [~,N] = size(points); % N��������
            
            %% �������
            code = obj.compute(points,obj.co_idx); % ִ���������
            if strcmp(option,'binary')
                c = code > repmat(obj.thresh,1,N);
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
            for d = 1:D
                center = learn.cluster.KMeansPlusPlus(code(d,:),2);
                obj.t(d) = sum(center)/2;
            end
            obj.t = reshape(obj.t,[],1);
        end
        
        %% ѵ��
        function obj = train(obj,points,labels,parameters)
            %% �������������
            if nargin <= 3
                parameters = [];
                disp('����train����ʱû�и�������������ʹ��Ĭ�ϲ�����');
            end
            
            %% ��ѵ������
            obj.points = points;
            obj.labels = labels;
            
            %% ��֯ѵ������
            labels_pos = labels(1:2,labels(3,:)==+1); % ���Ʊ�ǩ
            labels_neg = labels(1:2,labels(3,:)==-1); % ��ͬ��ǩ
            
            %% �������Ƽ���
            [~,N] = size(points);
            I = labels_pos(1,:); J = labels_pos(2,:);
            for n = 1:N
                idx = (J == n | I == n);
                obj.simset{n} = setdiff(union(I(idx),J(idx)),n);
            end
            
            %% ���㲻ͬ����
            I = labels_neg(1,:); J = labels_neg(2,:);
            for n = 1:N
                idx = (J == n | I == n);
                obj.difset{n} = setdiff(union(I(idx),J(idx)),n);
            end
            
            %% Ѱ��
            obj.weight = learn.optimal.minimize_sadam(obj,obj.weight,parameters);
            
            %% �����
            obj.points = [];
            obj.labels = [];
        end
    end
    
    methods(Static)
        function [] = unit_test()
            clear all;
            close all;
            rng(1);
            
            load('images.mat'); points = points(1:(32*32),:); points = double(points) / 255;
            load('labels_pos.mat');
            load('labels_neg.mat'); labels = [labels_pos labels_neg];
            
            configure = [1024,500,64,500,1024];
            nca = learn.ssc.NLNCA(configure,2,0.99);
            nca = nca.initialize();
            
            nca = nca.train(points,labels);
        end
    end
end

