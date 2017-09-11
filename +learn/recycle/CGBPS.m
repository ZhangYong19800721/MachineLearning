classdef CGBPS
    % Conjugate Gradient BP Sigmoid ��֪����ѵ����
    % ʹ��Conjugate Gradient BP�㷨ѵ��(�����ݶȷ��򴫲��㷨)
    
    properties
        points;
        labels;
        percep; 
    end
    
    methods % ���캯��
        function obj = CGBPS(points,labels,percep)
            obj.points = points;
            obj.labels = labels;
            obj.percep = percep;
        end
    end
    
    methods
        function y = object(obj,x)
            % ����Ŀ�꺯��
            obj.percep.weight = x;
            predict = obj.percep.do(obj.points);
            y = sum(sum((obj.labels - predict).^2));
        end
        
        function g = gradient(obj,x)
            %% ��ʼ��
            obj.percep.weight = x; % ��ʼ��Ȩֵ
            N = size(obj.points,2); % N��������
            P = size(obj.percep.weight,1); % P��������
            M = length(obj.percep.num_hidden); % M����
            g = zeros(P,N); % ÿһ�������������ݶ�
            
            %% �����ݶ�
            [~,a] = obj.percep.do(obj.points); % ִ���������
            s{M} = -2 * (obj.labels - a{M})' .* (a{M} .* (1 - a{M}))'; % ���㶥���������
            for m = (M-1):-1:1  % ���򴫲�������
                weight = obj.percep.getw(m+1);
                s{m} = s{m+1} * weight * diag(a{m}.*(1-a{m}));
            end
            
            parfor n = 1:N
                s = cell(1,M); % s������¼������
                s{M} = -2 * (obj.labels(:,n) - a{M})' .* (a{M} .* (1 - a{M}))'; % ���㶥���������
                for m = (M-1):-1:1  % ���򴫲�������
                    weight = obj.percep.getw(m+1);
                    s{m} = s{m+1} * weight * diag(a{m}.*(1-a{m}));
                end
                
                for m = 1:M
                    [~,cw] = obj.percep.getw(m);
                    [~,cb] = obj.percep.getb(m);
                    
                    H = obj.percep.num_hidden{m};
                    V = obj.percep.num_visual{m};
                    
                    if m == 1
                        f2w = repmat(s{m}',1,V) .* repmat(obj.points(:,n)',H,1);
                    else
                        f2w = repmat(s{m}',1,V) .* repmat(a{m-1}',H,1);
                    end
                    
                    g(cw,1) = g(cw,1) + reshape(f2w,[],1);
                    g(cb,1) = g(cb,1) + s{m}';
                end
            end
            
            g = g ./ N;
        end
    end
    
end

