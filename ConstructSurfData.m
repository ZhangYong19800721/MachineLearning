function SURF = ConstructSurfData(image_base_dir,file_type)
%CONSTRUCTSURFDATA 从给定的目录中读取所有指定类型图片的SURF特征
%   SURF特征集合
    files = dir(fullfile(image_base_dir,file_type)); % 得到所有图片的文件名
    N = length(files); % 图片文件的个数
    SURF = [];
    
    for n = 1:N
        image_rgb = imread(strcat(image_base_dir,files(n).name));
        image_yuv = rgb2ycbcr(image_rgb);
        points = detectSURFFeatures(image_yuv(:,:,1));
        [features,valid_points] = extractFeatures(image_yuv(:,:,1),points.selectStrongest(450));
        SURF = [SURF features'];
    end
end

