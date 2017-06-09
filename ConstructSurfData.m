function SURF = ConstructSurfData(image_base_dir,file_type)
%CONSTRUCTSURFDATA �Ӹ�����Ŀ¼�ж�ȡ����ָ������ͼƬ��SURF����
%   SURF��������
    files = dir(fullfile(image_base_dir,file_type)); % �õ�����ͼƬ���ļ���
    N = length(files); % ͼƬ�ļ��ĸ���
    SURF = [];
    
    for n = 1:N
        image_rgb = imread(strcat(image_base_dir,files(n).name));
        image_yuv = rgb2ycbcr(image_rgb);
        points = detectSURFFeatures(image_yuv(:,:,1));
        [features,valid_points] = extractFeatures(image_yuv(:,:,1),points.selectStrongest(450));
        SURF = [SURF features'];
    end
end

