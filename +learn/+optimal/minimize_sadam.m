function [x,y] = minimize_sadam(F,x0,parameters)
%minimize_adam Stochastic ADAM ����ݶ��½�
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
    
    if ~isfield(parameters,'window') % ������������û�и���window
        parameters.window = 1e+3; 
        disp(sprintf('û��window��������ʹ��Ĭ��ֵ%f',parameters.window));
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
    
    if ~isfield(parameters,'decay') % ������������û�и���decay
        parameters.decay = 0; % ȱʡ����²�����ѧϰ�ٶ�
        disp(sprintf('û��decay��������ʹ��Ĭ��ֵ%f',parameters.decay));
    end
    
    %% ��ʼ��
    T = parameters.max_it;
    W = parameters.window;
    beda1 = parameters.beda1;
    beda2 = parameters.beda2;
    r0 = parameters.learn_rate;
    D = parameters.decay;
    epsilon = parameters.epsilon;
    m = 0; % ��ʼ����һ����������
    v = 0; % ��ʼ���ڶ�����������
    x1 = x0;  % ��ʼ��
    y1 = F.object(x1,0); % ����Ŀ�꺯��ֵ
    
    %% ��ʼ����
    z = y1; k = inf; 
    for it = 1:T
        r  = r0 * exp(-D*it/T); 
        g1 = F.gradient(x1,it); % �����ݶ�
        m  = beda1 * m + (1 - beda1) * g1;    % ���µ�1����������
        v  = beda2 * v + (1 - beda2) * g1.^2; % ���µ�2����������
        mb  = m / (1 - beda1^it); % �Ե�1������������������
        vb  = v / (1 - beda2^it); % �Ե�2������������������
        x2 = x1 - r * mb ./ (sqrt(vb) + epsilon);
        y2 = F.object(x2,it); % ����Ŀ�꺯��ֵ
        z = (1-1/W)*z + (1/W)*y2;
        disp(sprintf('��������:%d ѧϰ�ٶ�:%f Ŀ���ֵ:%f Ŀ�꺯��:%f',it,r,z,y2));
        x1 = x2; y1 = y2;
        if mod(it,W) == 0
            if z < k
                k = z;
            else
                break;
            end
        end
    end
    
    %% ����
    x = x1;
    y = y1;
end

