function [a,b]=AR(F,x0,h0)
% AR advance & retreat ���˷�ȷ����������
% ���룺
%   F �������󣬵���F.object(x)����Ŀ�꺯��
%   x0 ��ʼ����λ��
%   h0 ��ʼ��������, h0 > 0
% �����
%   a ����������˵�
%   b ���������Ҷ˵�

    lamda = 1.0;
    x1 = x0;              F1 = F.object(x1); 
    x2 = x0 + lamda * h0; F2 = F.object(x2);
    if F1 > F2
        x = x2; Fx = F2;
        while true
            lamda = 2 * lamda; % ���󲽳�
            x1 = x2; 
            x2 = x1 + lamda * h0; 
            F2 = F.object(x2);
            if F2 > Fx
                break;
            end
        end
    else
        x = x1; Fx = F1;
        while true
            lamda = 2 * lamda; % ���󲽳�
            x1 = x2; 
            x2 = x1 - lamda * h0; 
            F2 = F.object(x2);
            if F2 > Fx
                break;
            end
        end
    end
    
    a=min(x,x2);
    b=max(x,x2);
end