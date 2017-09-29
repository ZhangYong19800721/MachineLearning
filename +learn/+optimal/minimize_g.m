function [x,y] = minimize_g(F,x0,parameters)
%minimize_g �ݶȷ�
%   ���룺
%       F ����F.gradient(x)����Ŀ�꺯�����ݶȣ�����F.object(x)����Ŀ�꺯����ֵ
%       x0 ��������ʼλ��
%       parameters ������
%   �����
%       x ���ŵĲ�����
%       y ��С�ĺ���ֵ

    %% ��������
    if nargin <= 2 % û�и�������
        parameters = [];
        disp('����minimize_g����ʱû�и�������������ʹ��Ĭ�ϲ�����');
    end
    
    if ~isfield(parameters,'epsilon') % ������������û�и���epsilon
        parameters.epsilon = 1e-3; 
        disp(sprintf('û��epsilon��������ʹ��Ĭ��ֵ%f',parameters.epsilon));
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
        parameters.learn_rate = 0.1;
        disp(sprintf('û��learn_rate��������ʹ��Ĭ��ֵ%f',parameters.learn_rate));
    end
    
    %% ��ʼ��
    m = parameters.momentum;
    r = parameters.learn_rate;
    inc_x = zeros(size(x0)); % �����ĵ�����
    x1 = x0;  
    y1 = F.object(x1); % ����Ŀ�꺯��ֵ
    
    %% ��ʼ����
    for it = 1:parameters.max_it
        g1 = F.gradient(x1); % �����ݶ�
        ng1 = norm(g1); % �����ݶ�ģ
        disp(sprintf('��������:%d ѧϰ�ٶ�:%f Ŀ�꺯��:%f �ݶ�ģ:%f ',it,parameters.learn_rate,y1,ng1));
        if ng1 < parameters.epsilon
            break; % ����ݶ��㹻С�ͽ�������
        end
        inc_x = m * inc_x - (1 - m) * r * g1; % ���ݶȷ����������ʹ�ö�������
        x1 = x1 + inc_x; % ���²���ֵ
        y1 = F.object(x1); % ����Ŀ�꺯��ֵ
    end
    
    %% ����
    x = x1;
    y = y1;
end

