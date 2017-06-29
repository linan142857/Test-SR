clear;clc;close all;
%% settings
folder = 'Test/Set5';
scale = 4;

%% generate data
filepaths = dir(fullfile(folder,'*.bmp'));
    
for i = 1 : length(filepaths)
    path = fullfile(folder,filepaths(i).name);

    image = imread(path);
    if(size(image, 3) == 3)
        image = rgb2ycbcr(image);
        image = image(:,:,1);
    end
    image = im2double(image);
    
    label = modcrop(image, scale);
    [hei,wid] = size(label);
    data = imresize(label,[hei/scale, wid/scale], 'bicubic');
    data = imresize(data,[hei,wid],'bicubic');
    label = reshape(label, [1, size(label, 1), size(label, 2)]);
    data = reshape(data, [1, size(data, 1), size(data, 2)]);
    save_name = regexp(filepaths(i).name, '.bmp', 'split');
    save_name = [save_name{1} '.mat'];
    save(['Set5/' save_name], 'data', 'label', '-v7.3');
end


