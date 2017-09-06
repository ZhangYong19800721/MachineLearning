classdef DAB_SSC_Pro_Aid
    %DAB_SSC_Pro_Aid DiscreteAdaBoostSSCPro�����Ż��ĸ�����
    %   �˴���ʾ��ϸ˵��
    
    properties
        weight;
        points;
        labels;
    end
    
    methods
        function obj = DAB_SSC_Pro_Aid(weight,points,labels)
            obj.weight = weight;
            obj.points = points;
            obj.labels = labels;
        end
        
        function g = gradient(obj,x)
            %% ��ʼ��
            [K,N] = size(obj.points); Q = K*K+K+1; % K���ݵ�ά�ȡ�N���ݵ�����Q���η��̲�������
            [A,B,C] = learn.tools.X2ABC(x); % ��x�ֽ�ΪA��B��C��������
            I = obj.labels(1,:); J = obj.labels(2,:); L = obj.labels(3,:);
            
            %% �����ݶ�
            f = learn.tools.quadratic(A,B,C,obj.points); % �������е��f����ֵ
            h = learn.tools.sigmoid(f); % �������е��h����ֵ
            g_f_C = ones(1,N); % f������C���ݶ�
            g_f_B = obj.points; % f������B���ݶ�
            g_f_A = 0.5 * obj.points(reshape(repmat(1:K,K,1),1,[]),:) .* obj.points(repmat(1:K,1,K),:); % f������A���ݶ�
            g_f_x = [g_f_A;g_f_B;g_f_C]; % f������x���ݶ�
            g_h_x = repmat(h.*(1-h),Q,1) .* g_f_x; % h������x���ݶ� 
            g_c_x = 4 * (g_h_x(:,I) .* repmat((h(J) - 0.5),Q,1)) + 4 * (g_h_x(:,J) .* repmat((h(I) - 0.5),Q,1)); % c������x���ݶ�
            g = sum(repmat(obj.weight.*L,Q,1) .* g_c_x,2);
        end
        
        function y = object(obj,x)
            %% ��ʼ��
            [A,B,C] = learn.tools.X2ABC(x); % ��x�ֽ�ΪA��B��C��������
            I = obj.labels(1,:); J = obj.labels(2,:); L = obj.labels(3,:);
            
            %% ����Ŀ�꺯��ֵ
            f = learn.tools.quadratic(A,B,C,obj.points); % �������е��f����ֵ
            h = learn.tools.sigmoid(f); % �������е��h����ֵ
            c = 4 * (h(I) - 0.5) .* (h(J) - 0.5); % �������е��c����ֵ
            y = sum(obj.weight .* L .* c,2); % ����Ŀ�꺯��ֵ
        end
    end
    
    methods(Static)
        function error_idx = unit_test1()
            clear all
            close all
            rng(2);
            [points,labels] = learn.data.GenerateData.type10(); [~,Q] = size(labels);
            weight = ones(1,Q) / Q;
            aid = learn.ssc.DAB_SSC_Pro_Aid(weight,points,labels);
            
            %% x0
            x0 = [2 0 0 2 0 0 -4.8^2]';
            [A,B,C] = learn.tools.X2ABC(x0); 
            f = @(x,y) 0.5*[x y]*A*[x y]' + B*[x y]' + C;
            ezplot(f,[-10,10,-10,10]);
            drawnow;
            y0 = aid.object(x0);
            g0 = aid.gradient(x0);
            
            %% x1
            x1 = x0 + 0.01 * g0;
            [A,B,C] = learn.tools.X2ABC(x1); 
            f = @(x,y) 0.5*[x y]*A*[x y]' + B*[x y]' + C;
            ezplot(f,[-10,10,-10,10]);
            drawnow;
            y1 = aid.object(x1);
            g1 = aid.gradient(x1);
            
            %% ����
            parameters.learn_rate = 0.1; % ѧϰ�ٶ�
            parameters.momentum = 0.9; % ���ٶ���
            parameters.epsilon = 1e-3; % ���ݶȵķ���С��epsilonʱ��������
            parameters.max_it = 1e4; % ����������
     
            x = learn.optimal.maximize_g(aid,x0,parameters);
                  
            [A,B,C] = learn.tools.X2ABC(x); 
            f = @(x,y) 0.5*[x y]*A*[x y]' + B*[x y]' + C;
            ezplot(f,[-10,10,-10,10]);
            drawnow;
            error_idx = 0;
        end
        
        function error_idx = unit_test2()
            clear all
            close all
            rng(2);
            points = [1 -1]; labels = [1 2 -1]'; 
            [~,Q] = size(labels);
            weight = ones(1,Q) / Q;
            aid = learn.ssc.DAB_SSC_Pro_Aid(weight,points,labels);
            
            %% x0
            x0 = [1 0 0]';
            [A,B,C] = learn.tools.X2ABC(x0); 
            f = @(x) 0.5*x*A*x'+B*x+C;
            ezplot(f,[-10,10,-10,10]);
            drawnow;
            y0 = aid.object(x0);
            g0 = aid.gradient(x0);
            
            %% x1
            x1 = x0 + 0.01 * g0;
            [A,B,C] = learn.tools.X2ABC(x1); 
            f = @(x) 0.5*x*A*x'+B*x+C;
            ezplot(f,[-10,10,-10,10]);
            drawnow;
            y1 = aid.object(x1);
            g1 = aid.gradient(x1);
            
            %% ����
            parameters.learn_rate = 0.01; % ѧϰ�ٶ�
            parameters.momentum = 0; % ���ٶ���
            parameters.epsilon = 1e-3; % ���ݶȵķ���С��epsilonʱ��������
            parameters.max_it = 2e3; % ����������
            x = learn.optimal.maximize_g(aid,x0,parameters);
            error_idx = 0;
        end
    end
    
end

