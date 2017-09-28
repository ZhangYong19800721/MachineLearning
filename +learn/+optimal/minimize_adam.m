function [x,y] = minimize_adam(F,x0,parameters)
%minimize_adam ADAM����ݶ��½�
%   �ο����ס�ADAM:A Method For Stochastic Optimization��,2014
%   ���룺
%       F ����F.gradient(x,i)����Ŀ�꺯�����ݶȣ�����F.object(x,i)����Ŀ�꺯����ֵ������iָʾminibatch
%       x0 ��������ʼλ��
%       parameters ������
%   �����
%       x ���ŵĲ�����
%       y ��С�ĺ���ֵ

    %% ��������
    if nargin <= 2 % û�и�������
        parameters = [];
        disp('����minimize_adam����ʱû�и�������������ʹ��Ĭ�ϲ�����');
    end
    
    if ~isfield(parameters,'epsilon') % ������������û�и���epsilon
        parameters.epsilon = 1e-8; 
        disp(sprintf('û��epsilon��������ʹ��Ĭ��ֵ%f',parameters.epsilon));
    end
    
    if ~isfield(parameters,'omiga') % ������������û�и���omiga
        parameters.omiga = 1e-6; 
        disp(sprintf('û��omiga��������ʹ��Ĭ��ֵ%f',parameters.omiga));
    end
    
    if ~isfield(parameters,'max_it') % ������������û�и���max_it
        parameters.max_it = 1e6;
        disp(sprintf('û��max_it��������ʹ��Ĭ��ֵ%d',parameters.max_it));
    end
    
    if ~isfield(parameters,'learn_rate') % ������������û�и���learn_rate
        parameters.learn_rate = 1e-3;
        disp(sprintf('û��learn_rate��������ʹ��Ĭ��ֵ%f',parameters.learn_rate));
    end
    
    if ~isfield(parameters,'beda1') 
        parameters.beda1 = 0.9;
        disp(sprintf('û��beda1��������ʹ��Ĭ��ֵ%f',parameters.beda1));
    end
    
    if ~isfield(parameters,'beda2') 
        parameters.beda2 = 0.999;
        disp(sprintf('û��beda2��������ʹ��Ĭ��ֵ%f',parameters.beda2));
    end
    
    %% ��ʼ��
    m = 0; % ��ʼ����һ����������
    v = 0; % ��ʼ���ڶ�����������
    x1 = x0;  % ��ʼ��
    y1 = F.object(x1); % ����Ŀ�꺯��ֵ
    
    %% ��ʼ����
    for it = 1:parameters.max_it
        g1 = F.gradient(x1,it); % �����ݶ�
        m  = parameters.beda1 * m + (1 - parameters.beda1) * g1;    % ���µ�1����������
        v  = parameters.beda2 * v + (1 - parameters.beda2) * g1.^2; % ���µ�2����������
        mb  = m / (1 - parameters.beda1^it); % �Ե�1������������������
        vb  = v / (1 - parameters.beda2^it); % �Ե�2������������������
        x2 = x1 - parameters.learn_rate * mb ./ (sqrt(vb) + parameters.epsilon);
        y2 = F.object(x2); % ����Ŀ�꺯��ֵ
        disp(sprintf('��������:%d ѧϰ�ٶ�:%f Ŀ�꺯��:%f ��������:%f ',it,parameters.learn_rate,y2,abs(y1-y2)));
        if abs(y1 - y2) < parameters.omiga
            x1 = x2; y1 = y2;
            break; % ����ݶ��㹻С�ͽ�������
        end
        x1 = x2; y1 = y2;
    end
    
    %% ����
    x = x1;
    y = y1;
end

