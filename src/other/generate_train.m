clear;clc;close all;
%% settings
folder = 'photo';
sz = 17;
scale = 4;
stride = 14;
%% initialization
data = zeros(5e5, sz, sz, 2);  
label = zeros(5e5, sz*scale, sz*scale, 1);
count = 0;

%% generate data
filepaths = dir(fullfile(folder,'*.png'));
    
for i = 1 : length(filepaths)
    path = fullfile(folder,filepaths(i).name);
    image = imread(path);
    if(size(image, 3) == 3)
        image = rgb2ycbcr(image);
        image = image(:,:,1);
    end
    image = im2double(image);
    im_label = modcrop(image, scale);
    [hei,wid] = size(im_label);
    im_input = imresize(im_label,[hei/scale, wid/scale], 'bicubic');
    [hei,wid] = size(im_input);
    for x = 1 : stride : hei-sz+1
        for y = 1 :stride : wid-sz+1

            subim_input = im_input(x : x+sz-1, y : y+sz-1);
            subim_label = im_label((x-1)*scale+1 : (x-1)*scale+sz*scale, (y-1)*scale+1 : (y-1)*scale+sz*scale);
            count=count+1;
            data(count, :, :, 1) = subim_input;
            label(count, :, :, 1) = subim_label;
        end
    end
    disp(count);
end

order = randperm(count);
cut = floor(0.9 * numel(order));
L = data(order, :, :, 1);
H = label(order, :, :, 1);
data = L(1:cut, :, :, 1);
label = H(1:cut, :, :, 1);
save('train.mat', 'data', 'label', '-v7.3');
data = L(cut+1:end, :, :, 1);
label = H(cut+1:end, :, :, 1);
save('vaild.mat', 'data', 'label', '-v7.3');
