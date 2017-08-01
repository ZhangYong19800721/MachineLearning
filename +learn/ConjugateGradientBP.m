classdef ConjugateGradientBP
    % ConjugateGradientBP ��֪����ѵ����
    % ʹ��Conjugate Gradient BP�㷨ѵ��(�����ݶȷ��򴫲��㷨)
    
    properties
        points;
        labels;
        percep; 
    end
    
    methods % ���캯��
        function obj = ConjugateGradientBP(points,labels,percep)
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
            obj.percep.weight = x;
            N = size(obj.points,2); % N��������
            P = size(obj.percep.weight,1); % P��������
            M = length(obj.percep.num_hidden); % M����
            g = zeros(P,1);
            
            for n = 1:N
                [~,a] = obj.percep.do(obj.points(:,n)); % ִ���������
                s = cell(1,M); % s������¼������
                s{M} = -2 * (obj.labels(:,n) - a{M})'; % ���㶥���������
                for m = (M-1):-1:1  % ���򴫲�������
                    weight = reshape(obj.percep.weight(obj.percep.star_w_idx{m+1}:obj.percep.stop_w_idx{m+1}),...
                                     obj.percep.num_hidden{m+1},...
                                     obj.percep.num_visual{m+1});
                    s{m} = s{m+1} * weight * diag(a{m}.*(1-a{m}));
                end
                
                for m = 1:M
                    cw = obj.percep.star_w_idx{m}:obj.percep.stop_w_idx{m};
                    cb = obj.percep.star_b_idx{m}:obj.percep.stop_b_idx{m};

                    H = obj.percep.num_hidden{m}; 
                    V = obj.percep.num_visual{m};
                    n2w = zeros(H,H*V);
                    for k = 1:H
                        z = zeros(H,V); 
                        if m == 1
                            z(k,:) = obj.points(:,n)';
                        else
                            z(k,:) = a{m-1}';
                        end
                        n2w(k,:) = reshape(z,1,[]);
                    end
                    g(cw,1) = g(cw,1) + (s{m} * n2w)';
                    
                    n2b = eye(H);
                    g(cb,1) = g(cb,1) + (s{m} * n2b)';
                end
            end
            
            g = g ./ N;
        end
    end
    
end

