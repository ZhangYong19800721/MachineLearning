classdef LMBPL
    % LMBPL ���ѵ��PerceptionL��֪����ѵ����
    % ʹ��Levenberg Marquardt BP�㷨
    
    properties
        points;
        labels;
        percep; 
    end
    
    methods % ���캯��
        function obj = LMBPL(points,labels,percep)
            obj.points = points;
            obj.labels = labels;
            obj.percep = percep;
        end
    end
    
    methods
        function y = vector(obj,x)
            obj.percep.weight = x;
            predict = obj.percep.do(obj.points);
            y = reshape(obj.labels - predict,[],1);
        end
        
        function j = jacobi(obj,x)
            obj.percep.weight = x;
            L = size(obj.labels,1); % L��ǩά��
            N = size(obj.points,2); % N��������
            P = size(obj.percep.weight,2); % P��������
            M = length(obj.percep.num_hidden); % M����
            j = zeros(L*N,P); % ��ʼ��jacobi����
            
            for n = 1:N
                [~,a] = obj.percep.do(obj.points(:,n)); % ִ���������
                
                s = cell(1,M);
                s{M} = -1 * eye(L); % ���㶥���������
                for m = (M-1):-1:1  % ���򴫲�������
                    weight = reshape(obj.percep.weight(obj.percep.star_w_idx{m+1}:obj.percep.stop_w_idx{m+1}),...
                                     obj.percep.num_hidden{m+1},...
                                     obj.percep.num_visual{m+1});
                    s{m} = s{m+1} * weight * diag(a{m}.*(1-a{m}));
                end
                
                r = ((n-1)*L+1):((n-1)*L+L);
                for m = 1:M
                    cw = obj.percep.star_w_idx{m}:obj.percep.stop_w_idx{m};
                    cb = obj.percep.star_b_idx{m}:obj.percep.stop_b_idx{m};

                    K = length(a{m}); 
                    H = obj.percep.num_hidden{m}; 
                    V = obj.percep.num_visual{m};
                    n2w = zeros(K,H*V);
                    for k = 1:K
                        g = zeros(H,V); 
                        if m == 1
                            g(k,:) = obj.points(:,n)';
                        else
                            g(k,:) = a{m-1}';
                        end
                        n2w(k,:) = reshape(g,1,[]);
                    end
                    j(r,cw) = s{m} * n2w;
                    
                    n2b = eye(K);
                    j(r,cb) = s{m} * n2b;
                end
            end
        end
    end
    
end

