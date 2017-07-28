function y = sample_softmax(x) 
    %SAMPLE_SOFTMAX softmax���� 
    %   x��ÿһ����һ��������������ÿһ��Ԫ�ش�����һ��softmax��Ԫ�ļ������
    
    [M,N] = size(x);           % ����ֵ�ĸ���
    y = zeros(size(x));        % y��ʼ��Ϊȫ0
    P = cumsum(x);             % �ۻ�����
    R = repmat(rand(1,N),M,1); % �����������
    D = 1 + sum(P<R); 
    I = sub2ind(size(x),D,1:N); 
    y(I) = 1;
end

