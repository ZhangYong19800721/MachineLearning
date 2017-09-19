classdef whiten
    %WHITEN ZCA Whitening Transform
    %   �ο���Learning Multiple Layers of Features from Tiny Images����Appendix A
    properties
        W;  % �׻�����
        D;  % ��׻�����
        A;  % ��ֵ����
    end

    methods
        function obj = compute(obj,X)
            [~,N] = size(X);
            obj.A = mean(X,2);
            X = double(X) - repmat(obj.A,1,N);
            [U,K,V] = svd(X*X');
            G = diag(K); G(G==0) = inf; G = 1./G; G = diag(G);
            obj.W =     sqrt(N-1) * V * sqrt(G) * U';
            obj.D = 1 / sqrt(N-1) * U * sqrt(K) * V';
        end
        
        function Y = white(X)
            Y = X - repmat(obj.A,1,size(X,2));
            Y = obj.W * Y;
        end
        
        function X = dewhite(Y)
            X = obj.D * Y;
            X = X + repmat(obj.A,1,size(X,2));
        end
    end
end

