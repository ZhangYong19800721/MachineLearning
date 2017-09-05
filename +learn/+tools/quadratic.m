function y = quadratic(A,B,C,x)
%quadratic ���ζ���ʽ������ʵ�� 
%   y = 0.5*x'*A*x+B*x+C
    [~,N] = size(x);
    y = 0.5*sum((x'*A).*x',2)' + B*x + repmat(C,1,N);
end

