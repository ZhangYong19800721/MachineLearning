% ������һ���Ƚ������Ķ�ȡ�ļ��������ݵĳ���

function [read_data] = batch_read_file(diretion, file_type)

% <<---�ļ������,����������--->>

%  ���Զ�ȡ���ļ���,������������ͬ��,�� nan

%  �����ļ���ֻ��������

%  *.xls �ļ�ֻ����Ӣ�Ļ���������

% ���������ݵĳ���, ֻ�ܶ�ȡ *.txt���ļ�, ���ļ������к�׺��

% [read_data_A]=batch_read_file('*.txt')

% [read_data_A]=batch_read_file('*.xls')

% file_typeΪ�ļ��ĸ�ʽ

% ����ֵ read_data ΪԪ����, ����ıȡ��������

% �� xls �ļ���˵��:

% xls�ļ����ļ�����������

% xls�ļ�ֻ������һ��������,���ж��,���ȡ��Ϊ����ֵ����һ��������

% file_type='*.txt'�� or file_type='*.xls';

%   ʾ��:�������ļ������������� 1.txt

%   1             2            3.345     34.522    12

%   1.2222     2.3333    3.4444

%   1             2            3            4            5     6     7     8     9

%   ������������һ�� (3,9)�ľ����ȱ�Ĳ���Ϊ nan



file_read=dir(fullfile(diretion,file_type));       % ���Ҫ��ȡ���ļ��б�

if strcmp(file_type,'*.xls');                         % ��ȡExcel�ļ�
    
    for i=1:length(file_read)
        
        file_name{i} = file_read(i).name;          % ��ȡ�ļ������б�
        
        % ��ʼ׼����� xls �ļ��ж�ȡ����
        
        read_data{i} = xlsread(file_name{i});
        
    end
    
else
    
    for i=1:length(file_read)                       % �ڴ˶ζ�ȡ *.txt ���ļ�
        
        file_name{i} = file_read(i).name;
        
        fid=fopen(file_name{i},'r');
        
        % ��ʼ׼������ļ��ж�ȡ����
        
        k=1;
        
        while ~feof(fid)
            
            temp=fgets(fid);                   % �����fgetl�ƺ�Ҳû�д�
            
            length_temp_data(k) = length(str2num(temp));    % �󳤶�
            
            temp_read_data{i}{k} = str2num(temp);            %   д����
            
            k=k+1;
            
        end        % end of for while ~
        
        max_row=max(length_temp_data);      % �����ĳ���,��ȷ���� nan �ĸ���
        
        for j=1:k-1
            
            len=length(temp_read_data{i}{j});
            
            if len<max_row
                
                temp_read_data{i}{j}(len:max_row) = nan;
                
            end
            
            read_data{i}(j,:) = temp_read_data{i}{j};
            
        end        % end of for j =
        
        fclose(fid);
        
    end  % end of for i =
    
end % end of if else