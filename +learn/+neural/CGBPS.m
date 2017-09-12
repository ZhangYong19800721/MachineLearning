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
        
        %% �ݶȼ���
        function g = gradient(obj,x)
            %% ��ʼ��
            obj.percep.weight = x; % ��ʼ��Ȩֵ
            N = size(obj.points,2); % N��������
            P = size(obj.percep.weight,1); % P��������
            M = length(obj.percep.num_hidden); % M����
            g = zeros(P,1); % �ݶ�
            s = cell(1,M); % ������
            w = cell(1,M); % Ȩֵ
            b = cell(1,M); % ƫ��
            for m = 1:M % ȡ��ÿһ���Ȩֵ��ƫ��ֵ
                w{m} = obj.percep.getw(m);
                b{m} = obj.percep.getb(m);
            end
            
            %% �����ݶ�
            [~,a] = obj.percep.do(obj.points); % ִ���������
            s{M} = -2 * (obj.labels - a{M})' .* (a{M} .* (1 - a{M}))'; % ���㶥���������
            for m = (M-1):-1:1  % ���򴫲�������
                sx = s{m+1}; wx = w{m+1}; ax = a{m}.*(1-a{m}); 
                sm = zeros(N,obj.percep.num_hidden{m});
                parfor n = 1:N
                    sm(n,:) = sx(n,:) * wx * diag(ax(:,n));
                end
                s{m} = sm;
            end
            
            for m = 1:M
                [~,cw] = obj.percep.getw(m);
                [~,cb] = obj.percep.getb(m);

                H = obj.percep.num_hidden{m};
                V = obj.percep.num_visual{m};
                
                sx = s{m}'; 
                if m == 1
                    px = obj.points';
                    gx = zeros(size(w{m}));
                    parfor n = 1:N
                        gx = gx + repmat(sx(:,n),1,V) .* repmat(px(n,:),H,1);
                    end
                else
                    ax = a{m-1}';
                    gx = zeros(size(w{m}));
                    parfor n = 1:N
                        gx = gx + repmat(sx(:,n),1,V) .* repmat(ax(n,:),H,1);
                    end
                end
                
                g(cw,1) = g(cw,1) + gx(:);
                g(cb,1) = g(cb,1) + sum(sx,2);
            end
            
            g = g ./ N;
        end
    end
    
end

