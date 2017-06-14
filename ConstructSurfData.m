function SURF = ConstructSurfData(image_base_dir,file_type)
%CONSTRUCTSURFDATA 从给定的目录中读取所有指定类型图片的SURF特征
%   SURF特征集合
    message = 'constructing surf data ....'; disp(message);
    files = dir(fullfile(image_base_dir,file_type)); % 得到所有图片的文件名
    N = length(files); % 图片文件的个数
    SURF = [];
    
    for n = 1:N
        read_image_file_number = n
        image_rgb = imread(strcat(image_base_dir,files(n).name));
        image_yuv = rgb2ycbcr(image_rgb);
        [~,width] = size(image_yuv(:,:,1));
        image_scale_down = imresize(image_yuv(:,:,1),320/width);
        points = detectSURFFeatures(image_scale_down);
        [features,~] = extractFeatures(image_scale_down,points.selectStrongest(450));
        SURF = [SURF features'];
    end
end

