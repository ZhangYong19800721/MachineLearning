% 下面是一个比较完整的读取文件夹里数据的程序

function [read_data] = batch_read_file(diretion, file_type)

% <<---文件已完成,程序已修正--->>

%  可以读取的文件中,若有列数不相同的,则补 nan

%  数据文件中只能是数字

%  *.xls 文件只能是英文或数字命名

% 批量读数据的程序, 只能读取 *.txt类文件, 且文件必须有后缀名

% [read_data_A]=batch_read_file('*.txt')

% [read_data_A]=batch_read_file('*.xls')

% file_type为文件的格式

% 返回值 read_data 为元数组, 保存谋取到的数据

% 对 xls 文件的说明:

% xls文件以文件名升序排序

% xls文件只能容许一个工作表,若有多个,则读取的为名字值最大的一个工作表

% file_type='*.txt'类 or file_type='*.xls';

%   示例:在数据文件中有如下内容 1.txt

%   1             2            3.345     34.522    12

%   1.2222     2.3333    3.4444

%   1             2            3            4            5     6     7     8     9

%   读出的数据是一个 (3,9)的矩阵空缺的部分为 nan



file_read=dir(fullfile(diretion,file_type));       % 获得要读取的文件列表

if strcmp(file_type,'*.xls');                         % 读取Excel文件
    
    for i=1:length(file_read)
        
        file_name{i} = file_read(i).name;          % 获取文件名的列表
        
        % 开始准备向从 xls 文件中读取数据
        
        read_data{i} = xlsread(file_name{i});
        
    end
    
else
    
    for i=1:length(file_read)                       % 在此段读取 *.txt 类文件
        
        file_name{i} = file_read(i).name;
        
        fid=fopen(file_name{i},'r');
        
        % 开始准备向从文件中读取数据
        
        k=1;
        
        while ~feof(fid)
            
            temp=fgets(fid);                   % 这儿用fgetl似乎也没有错
            
            length_temp_data(k) = length(str2num(temp));    % 求长度
            
            temp_read_data{i}{k} = str2num(temp);            %   写数据
            
            k=k+1;
            
        end        % end of for while ~
        
        max_row=max(length_temp_data);      % 求最大的长度,以确定补 nan 的个数
        
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