function [x,y] = minimize_sgd(F,x0,parameters)
%minimize_sgd ����ݶ��½�
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
        disp('����minimize_sgd����ʱû�и�������������ʹ��Ĭ�ϲ�����');
    end
    
    if ~isfield(parameters,'window') % ������������û�и���window
        parameters.window = 1e+3; 
        disp(sprintf('û��window��������ʹ��Ĭ��ֵ%f',parameters.window));
    end
    
    if ~isfield(parameters,'max_it') % ������������û�и���max_it
        parameters.max_it = 1e6;
        disp(sprintf('û��max_it��������ʹ��Ĭ��ֵ%d',parameters.max_it));
    end
    
    if ~isfield(parameters,'momentum') % ������������û�и���momentum
        parameters.momentum = 0.9;
        disp(sprintf('û��momentum��������ʹ��Ĭ��ֵ%f',parameters.momentum));
    end
    
    if ~isfield(parameters,'learn_rate') % ������������û�и���learn_rate
        parameters.learn_rate = 0.01;
        disp(sprintf('û��learn_rate��������ʹ��Ĭ��ֵ%f',parameters.learn_rate));
    end
    
    if ~isfield(parameters,'decay') % ������������û�и���decay
        parameters.decay = 1; % ȱʡ����²�����ѧϰ�ٶ�
        disp(sprintf('û��decay��������ʹ��Ĭ��ֵ%f',parameters.decay));
    end
    
    %% ��ʼ��
    inc_x = zeros(size(x0)); % �����ĵ�����
    m = parameters.momentum;
    r0 = parameters.learn_rate;
    D = parameters.decay;
    T = parameters.max_it;
    W = parameters.window;
    x1 = x0;  
    y1 = F.object(x1,0); % ����Ŀ�꺯��ֵ
    
    %% ��ʼ����
    z = 100 * y1; k = inf; 
    for it = 1:T
        r  = r0 - (1 - 1/D) * r0 * it / T;
        g1 = F.gradient(x1,it); % �����ݶ�
        inc_x = m * inc_x - (1 - m) * r * g1; % ���ݶȷ����������ʹ�ö�������
        x1 = x1 + inc_x; % ���²���ֵ
        y1 = F.object(x1,it); % ����Ŀ�꺯��ֵ
        z = (1-1/W)*z + (1/W)*y1;
        disp(sprintf('��������:%d ѧϰ�ٶ�:%f ������ֵ:%f',it,r,z));
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

