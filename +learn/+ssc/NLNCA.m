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
            point_a = obj.points(:,a); % ȡ��a����
            point_b = obj.points(:,obj.simset{a}); % ȡ���Ƶ㼯
            point_z = obj.points(:,obj.difset{a}); % ȡ��ͬ�㼯
            
            %% ����ʵ������
            code_a = obj.encode(point_a);
            code_b = obj.encode(point_b);
            code_z = obj.encode(point_z);
            
            %% a�㵽�������е�ľ���
            dis_ab = sum((code_b - repmat(code_a,1,size(code_b,2))).^2,1); 
            dis_az = sum((code_z - repmat(code_a,1,size(code_z,2))).^2,1); 
            exp_ab = exp(-dis_ab); % a�㵽���Ƶ����ĸ�ָ������
            exp_az = exp(-dis_az); % a�㵽��ͬ�����ĸ�ָ������
            sum_ex = sum([exp_ab exp_az]);
            
            %% a�㵽���Ƶ�ĸ���
            p_ab = exp_ab / sum_ex;
            
            %% ����Ŀ�꺯��
            y = obj.lamdax * sum(p_ab) + (1 - obj.lamdax) * cross_entropy;
        end
        
        %% �����ݶ�
        function g = gradient(obj,x,a)
            %% ��ʼ��
            obj.weight = x; % �������ò���
            point_a = obj.points(:,a); % ȡ��a����
            point_b = obj.points(:,obj.simset{a}); % ȡ���Ƶ㼯
            point_z = obj.points(:,obj.difset{a}); % ȡ��ͬ�㼯
            
            %% ����ʵ������
            code_a = obj.encode(point_a);
            code_b = obj.encode(point_b);
            code_z = obj.encode(point_z);
            
            %% a�㵽�������е�ľ���
            dis_ab = sum((code_b - repmat(code_a,1,size(code_b,2))).^2,1); 
            dis_az = sum((code_z - repmat(code_a,1,size(code_z,2))).^2,1); 
            exp_ab = exp(-dis_ab); % a�㵽���Ƶ����ĸ�ָ������
            exp_az = exp(-dis_az); % a�㵽��ͬ�����ĸ�ָ������
            sum_ex = sum([exp_ab exp_az]);
            
            %% a�㵽�������е�ĸ���
            p_ab = exp_ab / sum_ex; % a�㵽���Ƶ�ĸ���
            p_az = exp_az / sum_ex; % a�㵽��ͬ��ĸ���
            p_a  = sum(p_ab); % a�㵽���Ƶ�ĸ��ʺ�
            
            %% ���㶥���������
            
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
    end
    
end

