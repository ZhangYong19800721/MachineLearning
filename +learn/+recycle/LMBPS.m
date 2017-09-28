classdef LMBPS
    % LMBPS ���ѵ��PerceptionS��֪����ѵ����
    % ʹ��Levenberg Marquardt BP�㷨
    
    properties
        points;
        labels;
        percep; 
    end
    
    methods % ���캯��
        function obj = LMBPS(points,labels,percep)
            obj.points = points;
            obj.labels = labels;
            obj.percep = percep;
        end
    end
    
    methods
        function y = object(obj,x)
            obj.percep.weight = x;
            predict = obj.percep.do(obj.points);
            y = sum(sum((obj.labels - predict).^2));
        end
        
        function [H,G] = hessen(obj,x)
            %% ��ʼ��
            obj.percep.weight = x; % ���ò���
            [K,~] = size(obj.labels); % K��ǩά��
            [~,N] = size(obj.points); % N��������
            P = numel(obj.percep.weight); % P��������
            M = length(obj.percep.num_hidden); % M����
            H = zeros(P,P); % Hessian����
            G = zeros(P,1); % �ݶ�����
            s = cell(1,M); % ������
            w = cell(1,M); cw = cell(1,M); % Ȩֵ
            b = cell(1,M); cb = cell(1,M); % ƫ��
            for m = 1:M % ȡ��ÿһ���Ȩֵ��ƫ��ֵ
                [w{m},cw{m}] = obj.percep.getw(m);
                [b{m},cb{m}] = obj.percep.getb(m);
            end
            
            %% ����Jacobi����Hessian����Gradient�ݶ�
            [~,a] = obj.percep.do(obj.points); % ִ���������
            for n = 1:N
                %% ���򴫲�������
                s{M} = diag(-a{M}(:,n).*(1-a{M}(:,n))); % ���㶥���������
                for m = (M-1):-1:1  % ���򴫲�
                    s{m} = s{m+1} * w{m+1} * diag(a{m}(:,n).*(1-a{m}(:,n)));
                end
                
                %% ����Jacobi����
                J = zeros(K,P);
                for m = 1:M
                    if m == 1
                        Jw = s{m} * kron(eye(obj.percep.num_hidden{m}),obj.points(:,n)'); % �����Զ�Ȩֵ�ĵ���
                        Jb = s{m}; 
                    else
                        Jw = s{m} * kron(eye(obj.percep.num_hidden{m}),a{m-1}(:,n)'); % �����Զ�Ȩֵ�ĵ��� 
                        Jb = s{m};
                    end
                    J(:,cw{m}) = Jw;
                    J(:,cb{m}) = Jb;
                end
                
                %% ����Hessian����Gradient�ݶ�
                H = H + J'*J;
                G = G + 2*J'*(obj.labels(:,n) - a{M}(:,n));
            end
        end
    end
end

