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
    
    if ~isfield(parameters,'epsilon') % ������������û�и���epsilon
        parameters.epsilon = 1e-3; 
        disp(sprintf('����minimize_sgd����ʱ��������û��epsilon��������ʹ��Ĭ��ֵ%f',parameters.epsilon));
    end
    
    if ~isfield(parameters,'max_it') % ������������û�и���max_it
        parameters.max_it = 1e6;
        disp(sprintf('����minimize_sgd����ʱ��������û��max_it��������ʹ��Ĭ��ֵ%d',parameters.max_it));
    end
    
    if ~isfield(parameters,'momentum') % ������������û�и���momentum
        parameters.momentum = 0.9;
        disp(sprintf('����minimize_sgd����ʱ��������û��momentum��������ʹ��Ĭ��ֵ%f',parameters.momentum));
    end
    
    if ~isfield(parameters,'learn_rate') % ������������û�и���learn_rate
        parameters.learn_rate = 0.1;
        disp(sprintf('����minimize_sgd����ʱ��������û��learn_rate��������ʹ��Ĭ��ֵ%f',parameters.learn_rate));
    end
    
    if ~isfield(parameters,'decay') % ������������û�и���decay
        parameters.decay = 0; % ȱʡ����²�����ѧϰ�ٶ�
        disp(sprintf('����minimize_sgd����ʱ��������û��decay��������ʹ��Ĭ��ֵ%f',parameters.decay));
    end
    
    %% ��ʼ��
    inc_x = zeros(size(x0)); % �����ĵ�����
    m = parameters.momentum;
    x1 = x0;  
    
    %% ��ʼ����
    for it = 0:parameters.max_it
        r  = parameters.learn_rate / (1 + parameters.decay * it / parameters.max_it);
        g1 = F.gradient(x1,it); % �����ݶ�
        y1 = F.object(x1,it); % ����Ŀ�꺯��ֵ
        ng1 = norm(g1); % �����ݶ�ģ
        disp(sprintf('��������:%d ѧϰ�ٶ�:%f Ŀ�꺯��:%f �ݶ�ģ:%f ',it,r,y1,ng1));
        if ng1 < parameters.epsilon
            break; % ����ݶ��㹻С�ͽ�������
        end
        inc_x = m * inc_x - (1 - m) * r * g1; % ���ݶȷ����������ʹ�ö�������
        x1 = x1 + inc_x; % ���²���ֵ
    end
    
    %% ����
    x = x1;
    y = y1;
end

