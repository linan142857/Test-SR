clear;clc;close all;
%% settings
folder = 'photo-t/Turbine';
scale = 4;

%% generate data
filepaths = dir(fullfile(folder,'*.png'));
image = imread(fullfile(folder,'50.png'));
image = modcrop(image, scale);
[hei,wid,~] = size(image);
data = zeros(length(filepaths), hei/scale, wid/scale);
label = zeros(length(filepaths), hei, wid);
for i = 1 : length(filepaths)
    path = fullfile(folder,filepaths(i).name);
    image = imread(path);
    if(size(image, 3) == 3)
        image = rgb2ycbcr(image);
        image = image(:,:,1);
    end
    image = im2double(image);
    image = modcrop(image, scale);
    label(i,:,:) = image;
    data(i, :, :) = imresize(image,[hei/scale, wid/scale], 'bicubic');
end
save('Turbine.mat', 'data', 'label', '-v7.3');


