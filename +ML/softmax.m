function y = softmax(x)
%SOFTMAX ����softmax��Ԫ�ļ������
%   
    N = size(x,1);
    y = exp(x);
    y = y ./ repmat(sum(y),N,1);
end

