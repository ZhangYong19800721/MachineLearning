function [A,B,C] = X2ABC(x)
    Q = numel(x); K = round((sqrt(4*Q-3)-1)/2);
    A = reshape(x(1:(K*K)),K,K); B = reshape(x(K*K+(1:K)),1,[]); C = x(end);
end