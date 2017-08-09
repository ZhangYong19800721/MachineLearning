function [W,D,A] = whiten(X)
%WHITEN ZCA Whitening Transform
%   ²Î¿¼¡°Learning Multiple Layers of Features from Tiny Images¡±µÄAppendix A
    [~,N] = size(X);
    A = mean(X,2);
    X = double(X) - repmat(A,1,N);
    [U,K,V] = svd(X*X');
    G = diag(K); G(G==0) = inf; G = 1./G; G = diag(G);
    W =     sqrt(N-1) * V * sqrt(G) * U'; 
    D = 1 / sqrt(N-1) * U * sqrt(K) * V';
end

