clear;clc;close all;
%% settings
folder = '291';
sz = 41;
scale = 4;
stride = 14;
%% initialization
data = single(zeros(5e5, sz, sz, 3));
label = single(zeros(5e5, sz, sz, 3));
count = 0;
aug = 0;
G = fspecial('gaussian', [5, 5], 2);
%% generate data
filepaths = dir(fullfile(folder));
    
for i = 3 : length(filepaths)
    path = fullfile(folder,filepaths(i).name);
    image = imread(path);
    if(size(image, 3) == 1)
        continue;
        % image = rgb2ycbcr(image);
        % image = image(:,:,1);
    end
    image = im2single(image);
    im_label = modcrop(image, scale);
    if aug == 0
        [hei, wid, ~] = size(im_label);
        im_input = imresize(im_label,1/scale, 'bicubic');
        im_input = imresize(im_input,scale,'bicubic');
        for x = 1 : stride : hei-sz+1
            for y = 1 :stride : wid-sz+1
                subim_input = im_input(x : x+sz-1, y : y+sz-1, :);
                subim_label = im_label(x : x+sz-1, y : y+sz-1, :);

                count=count+1;
                data(count, :, :, :) = subim_input;
                label(count, :, :, :) = subim_label;
            end
        end
    else
        im_inputs = cell(8,1);
        im_inputs{1} = im_label;
        im_inputs{2} = imrotate(im_label, 90);
        im_inputs{3} = imrotate(im_label, 180);
        im_inputs{4} = imrotate(im_label, 270);
        im_inputs{5} = flipud(im_label);
        im_inputs{6} = fliplr(im_label);
        im_inputs{7} = imrotate(flipud(im_label), 90);
        im_inputs{8} = imrotate(fliplr(im_label), 90);
        for k = 1:8
            im_label = im_inputs{k};
            [hei,wid] = size(im_label);
            im_input = imresize(im_label,[hei/scale, wid/scale], 'bicubic');
            im_input = imresize(im_input,[hei,wid],'bicubic');

            for x = 1 : stride : hei-sz+1
                for y = 1 :stride : wid-sz+1

                    subim_input = im_input(x : x+sz-1, y : y+sz-1);
                    subim_label = im_label(x : x+sz-1, y : y+sz-1);

                    count=count+1;
                    data(count, :, :, 1) = subim_input;
                    label(count, :, :, 1) = subim_label;
                end
            end
        end
    end
    disp(count);
end

order = randperm(count);
cut = floor(0.9 * numel(order));
L = single(data(order, :, :, :));
H = single(label(order, :, :, :));
data = L(1:cut, :, :, :);
label = H(1:cut, :, :, :);
save('BSR Train+91-train.mat', 'data', 'label', '-v7.3');
data = L(cut+1:end, :, :, :);
label = H(cut+1:end, :, :, :);
save('BSR Train+91-vaild.mat', 'data', 'label', '-v7.3');
