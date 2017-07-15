function y = sample_softmax(x) 
%SAMPLE_SOFTMAX ����softmax����������x��ÿһ����һ��������������ÿһ��Ԫ�ش�����һ��softmax��Ԫ�ļ������
%   
    [row col] = size(x);
    accumulate = zeros(1,col);
    for r = 1:row
        accumulate = accumulate + x(r,:);
        x(r,:) = accumulate;
    end
    
    z = rand(1,col);
    z = repmat(z,row,1);
    y = (x>z);
    k = zeros(size(y));
    k(2:row,:) = y(1:(row-1),:);
    y = y - k;
end

