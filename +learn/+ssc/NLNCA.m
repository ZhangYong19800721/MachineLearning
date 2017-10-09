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
            point_a = obj.points(:,a); Na = 1; % ȡ��a����
            point_b = obj.points(:,obj.simset{a}); Nb = numel(obj.simset{a}); % ȡ���Ƶ㼯
            point_z = obj.points(:,obj.difset{a}); Nz = numel(obj.difset{a}); % ȡ��ͬ�㼯
            
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
            e_a = point_a .* log(y_a) + (1-point_a) .* log(1-y_a);
            e_b = point_b .* log(y_b) + (1-point_b) .* log(1-y_b);
            e_z = point_z .* log(y_z) + (1-point_z) .* log(1-y_z);
            c_e = -sum(sum([e_a e_b e_z])) / (Na+Nb+Nz);
            
            %% ����Ŀ�꺯��
            % y = obj.lamdax * p_a + (1 - obj.lamdax) * c_e;
            y = p_a;
        end
        
        %% �����ݶ�
        function g = gradient(obj,x,a)
            %% ��ʼ��
            obj.weight = x; % �������ò���
            L = length(obj.num_hidden); % L����
            g = zeros(size(obj.weight,1),1); % �ݶ�
            point_a = obj.points(:,a); % ȡ��a����
            point_b = obj.points(:,obj.simset{a}); % ȡ���Ƶ㼯
            point_z = obj.points(:,obj.difset{a}); % ȡ��ͬ�㼯
            s = cell(1,L); % ������
            w = cell(1,L); iw = cell(1,L); % Ȩֵ
            b = cell(1,L); ib = cell(1,L); % ƫ��
            for l = 1:L % ȡ��ÿһ���Ȩֵ��ƫ��ֵ
                [w{l},iw{l}] = obj.getw(l);
                [b{l},ib{l}] = obj.getb(l);
            end
            
            %% ����ʵ������
            [~,f_a] = obj.do(point_a); c_a = f_a{obj.co_idx};
            [~,f_b] = obj.do(point_b); c_b = f_b{obj.co_idx};
            [~,f_z] = obj.do(point_z); c_z = f_z{obj.co_idx};
            c_ab = repmat(c_a,1,size(c_b,2)) - c_b;
            c_az = repmat(c_a,1,size(c_z,2)) - c_z;
            
            %% a�㵽�������е�ľ���
            dis_ab = sum(c_ab.^2,1); 
            dis_az = sum(c_az.^2,1); 
            exp_ab = exp(-dis_ab); % a�㵽���Ƶ����ĸ�ָ������
            exp_az = exp(-dis_az); % a�㵽��ͬ�����ĸ�ָ������
            sum_ex = sum([exp_ab exp_az]);
            
            %% a�㵽�������е�ĸ���
            p_ab = exp_ab / sum_ex; % a�㵽���Ƶ�ĸ���
            p_az = exp_az / sum_ex; % a�㵽��ͬ��ĸ���
            p_a  = sum(p_ab); % a�㵽���Ƶ�ĸ��ʺ�
            
            %% ���㶥���������
            s_a = c_a .* (1-c_a);
            s_b = c_b .* (1-c_b);
            s_z = c_z .* (1-c_z);
            s_ab = repmat(s_a,1,size(s_b,2)) - s_b;
            s_az = repmat(s_a,1,size(s_z,2)) - s_z;
            s1 = sum(repmat(p_az,size(c_az,1),1) .* (2 * c_az .* s_az),2);
            s2 = sum(repmat(p_ab,size(c_ab,1),1) .* (2 * c_ab .* s_ab),2);
            s{obj.co_idx} = p_a * (s1+s2) - s2;
            
            %% ���򴫲�������
            for l = L:-1:1  
                if l < obj.co_idx
                    s{l} = (s{l+1} * w{l+1}) .* (a{l}.*(1-a{l}))';
                elseif l > obj.co_idx
                    H = obj.num_hidden{l};
                    s{l} = zeros(1,H);
                end
            end
            
            for l = 1:L
                H = obj.num_hidden{l};
                V = obj.num_visual{l};
                if l == 1
                    minibatch = [point_a point_b point_z];
                    gx = reshape(repmat(s{l}',V,1),H,V,S) .* reshape(repelem(minibatch,H,1),H,V,S);
                elseif l <= obj.co_idx
                    minibatch = [f_a{l-1} f_b{l-1} f_z{l-1}];
                    gx = reshape(repmat(s{l}',V,1),H,V,S) .* reshape(repelem(minibatch,H,1),H,V,S);     
                else
                    gx = zeros(H,V);
                end
                gx = sum(gx,3);
                g(iw{l},1) = g(iw{l},1) + gx(:);
                g(ib{l},1) = g(ib{l},1) + sum(s{l},1)';
            end
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
            parfor n = 1:N
                idx = (J == n | I == n);
                simset{n} = setdiff(union(I(idx),J(idx)),n);
            end
            obj.simset = simset;
            
            %% ���㲻ͬ����
            I = labels_neg(1,:); J = labels_neg(2,:);
            parfor n = 1:N
                idx = (J == n | I == n);
                difset{n} = setdiff(union(I(idx),J(idx)),n);
            end
            obj.difset = difset;
            
            %% Ѱ��
            obj.weight = learn.optimal.maximize_sadam(obj,obj.weight,parameters);
            
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
            load('labels_neg.mat'); 
            labels = [labels_pos labels_neg];
            
            configure = [1024,500,64,500,1024];
            nca = learn.ssc.NLNCA(configure,2,0.99);
            nca = nca.initialize();
            
            nca = nca.train(points,labels);
        end
    end
end

