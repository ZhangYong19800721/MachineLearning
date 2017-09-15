function [a,b]=ARR(F,x0,h0,parameters)
% ARR �Ľ��Ľ��˷�ȷ��������������
% ���룺
%   F �������󣬵���F.object(x)����Ŀ�꺯��
%   x0 ��ʼ����λ��
%   h0 ��ʼ����������h0��������������������
% �����
%   a ����������˵�
%   b ���������Ҷ˵�

    %% ��������
    if nargin <= 3 % û�и������������
        parameters = [];
        % disp('����gold����ʱû�и�������������ʹ��Ĭ�ϲ���');
    end
    
    if ~isfield(parameters,'epsilon') % ������������û�и���epsilon�����
        parameters.epsilon = 1e-6; 
        % disp(sprintf('����ARR����ʱ��������û��epsilon��������ʹ��Ĭ��ֵ%f',parameters.epsilon));
    end

    %%
    F0 = F.object(x0); 
    x1 = x0 + h0; F1 = F.object(x1);
    
    if F0 > F1
        while F0 > F1
            h0 = 2 * h0; % ���󲽳�
            x1 = x0 + h0; F1 = F.object(x1);
        end
        a = x0;
        b = x1;
    else
        a = x0;
        while F0 <= F1
            h0 = 0.5 * h0; % ��С����
            if abs(h0) < parameters.epsilon
                break;
            end
            b  = x1;
            x1 = x0 + h0; F1 = F.object(x1);
        end
    end
end