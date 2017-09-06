function y = quadratic(A,B,C,x)
%quadratic ���ζ���ʽ������ʵ��
%   ���룺
%       x ÿһ����һ�����ݵ�
%   �����
%       y = 0.5*x'*A*x+B*x+C ���κ�������Ϊx����������ݵ㣬����ʹ������ļ��㹫ʽ
%   
    [~,N] = size(x);
    y = 0.5*sum((x'*A).*x',2)' + B*x + repmat(C,1,N);
end

