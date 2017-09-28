function [x1,y1] = minimize_cg(F,x0,parameters)
%minimize_cg �����ݶȷ�
% ���룺
%   F �������󣬵���F.object(x)����Ŀ�꺯����x����ֵ������F.gradient(x)����Ŀ�꺯����x�����ݶ�
%   x0 ��������ʼλ��
%   parameters.epsilon ���ݶ�ģС��epsilonʱֹͣ����
%   parameters.max_it ����������
%   parameters.reset ��������
%   parameters.option ���������� 
%       'gold' �ƽ�ָ����ȷ������
%       'parabola' ����������������ȷ������
%       'armijo' Armijo׼�򣨷Ǿ�ȷ������
%   �����ûƽ�ָ��������ʱ��
%       parameters.gold.epsilon ����������ֹͣ����
%   ������Armijo��������ʱ��
%       parameters.armijo.beda ֵ������(0,  1)֮�䣬����ֵ0.5
%       parameters.armijo.alfa ֵ������(0,0.5)֮�䣬����ֵ0.2
%       parameters.armijo.maxs �������������������������ֵ30
%   �����������߷���������ʱ��
%       parameters.parabola.epsilon ����������ֹͣ����

    %% ��������
    if nargin <= 2 % û�и�������
        parameters = [];
        disp('����minimize_cg����ʱû�и�������������ʹ��Ĭ�ϲ�����');
    end
    
    if ~isfield(parameters,'epsilon') % ������������û�и���epsilon
        parameters.epsilon = 1e-3; 
        disp(sprintf('����minimize_cg����ʱ��������û��epsilon��������ʹ��Ĭ��ֵ%f',parameters.epsilon));
    end
    
    if ~isfield(parameters,'max_it') % ������������û�и���max_it
        parameters.max_it = 1e6;
        disp(sprintf('����minimize_cg����ʱ��������û��max_it��������ʹ��Ĭ��ֵ%d',parameters.max_it));
    end
    
    if ~isfield(parameters,'reset') % ������������û�и���reset
        parameters.reset = 500;
        disp(sprintf('����minimize_cg����ʱ��������û��reset��������ʹ��Ĭ��ֵ%f',parameters.reset));
    end
    
    if ~isfield(parameters,'option') % ������������û�и���option
        parameters.option = 'gold';
        disp(sprintf('����minimize_cg����ʱ��������û��option��������ʹ��Ĭ��ֵ%s',parameters.option));
    end
    
    if ~isfield(parameters,'gold') % ������������û�и���gold
        parameters.gold = [];
    end
    
    if ~isfield(parameters,'parabola') % ������������û�и���parabola
        parameters.parabola = [];
    end
    
    if ~isfield(parameters,'armijo') % ������������û�и���armijo
        parameters.armijo = [];
    end

    %% ������ʼλ�õĺ���ֵ���ݶȡ��ݶ�ģ
    x1 = x0; y1 = F.object(x1); g1 = F.gradient(x1); ng1 = norm(g1); % ��ʼ��Ϊx0,���㺯��ֵ���ݶȡ��ݶ�ģ 
    if ng1 < parameters.epsilon, return; end % ����ݶ��㹻С��ֱ�ӷ���
    
    %% ����Ѱ��
    d1 = -g1; % ��ʼ��������Ϊ���ݶȷ���
    for it = 1:parameters.max_it
        if ng1 < parameters.epsilon, return; end % ����ݶ��㹻С��ֱ�ӷ���
        
        % ��d1����������
        if strcmp(parameters.option,'gold') % �ƽ�ָ����һά��ȷ������
            Fs = learn.optimal.SINGLEX(F,x1,d1); % ��װΪ����������
            [a,b] = learn.optimal.ARR(Fs,0,1,parameters.gold); % ȷ����������
            [y2,lamda] = learn.optimal.gold(Fs,a,b,parameters.gold); x2 = x1 + lamda * d1;
        elseif strcmp(parameters.option,'parabola') % ʹ�������߷�����һά����
            Fs = learn.optimal.SINGLEX(F,x1,d1); % ��װΪ����������
            [a,b] = learn.optimal.ARR(Fs,0,1,parameters.parabola); % ȷ����������
            [y2,lamda] = learn.optimal.parabola(Fs,a,b,parameters.parabola); x2 = x1 + lamda * d1;
        elseif strcmp(parameters.option,'armijo') % armijo׼�����һά�Ǿ�ȷ����
            [y2,x2] = learn.optimal.armijo(F,x1,g1,d1,parameters.armijo);
        end
        
        c1 = mod(it,parameters.reset) == 0; % �������õ�
        c2 = y1 <= y2; %����d1������һ���½�����
        if c1 || c2
            d1 = -g1; % �趨��������Ϊ���ݶȷ���
            if strcmp(parameters.option,'gold') % �ƽ�ָ����һά��ȷ������
                Fs = learn.optimal.SINGLEX(F,x1,d1); % ��װΪ����������
                [a,b] = learn.optimal.ARR(Fs,0,1,parameters.gold); % ȷ����������
                [y2,lamda] = learn.optimal.gold(Fs,a,b,parameters.gold); x2 = x1 + lamda * d1;
            elseif strcmp(parameters.option,'parabola') % ʹ�������߷�����һά����
                Fs = learn.optimal.SINGLEX(F,x1,d1); % ��װΪ����������
                [a,b] = learn.optimal.ARR(Fs,0,1,parameters.parabola); % ȷ����������
                [y2,lamda] = learn.optimal.parabola(Fs,a,b,parameters.parabola); x2 = x1 + lamda * d1;
            elseif strcmp(parameters.option,'armijo') % armijo׼�����һά�Ǿ�ȷ����
                [y2,x2] = learn.optimal.armijo(F,x1,g1,d1,parameters.armijo);
            end
            g2 = F.gradient(x2); d2 = -g2; ng2 = norm(g2); % �������µ�λ��x2�������㺯��ֵ���ݶȡ����������ݶ�ģ
            x1 = x2; d1 = d2; g1 = g2; y1 = y2; ng1 = ng2;
            disp(sprintf('Ŀ�꺯��:%f ��������:%d �ݶ�ģ:%f ',y1,it,ng1));
            continue;
        end
 
        g2 = F.gradient(x2); ng2 = norm(g2); % ����x2�����ݶȺ��ݶ�ģ
        beda = g2'*(g2-g1)/(g1'*g1); d2 = -g2 + beda * d1; % ����x2������������d2
        x1 = x2; d1 = d2; g1 = g2; y1 = y2; ng1 = ng2;
        disp(sprintf('Ŀ�꺯��:%f ��������:%d �ݶ�ģ:%f ',y1,it,ng1));
    end
end

