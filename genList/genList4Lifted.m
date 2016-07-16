clear; clc;
% load feature matrix
load('feature.mat');  

% initial setting
num_img = size(feature, 1);
num_positives = 64;
num_negtives = 64;

% write into .txt file setting
filename = 'train_batch128.txt';
rootpath = '/path/to/the/root/of/dataset/';
gen_file = fopen(filename, 'wt');

% pre-processing raw image list .txt file in order to
% generate labels(ground-truth) corresponding the index
train = importdata('train.txt');
label = train.data;
path = train.textdata;

% calculate the distance matrix
disp('Starting distance matrix calculation')
distmatrix = pdist2(feature, feature, 'cosine');
disp('distance matrix calculation finished')

% rawidx indicate the index of image feature or current index for processing 
for rawidx = 1:num_img
    % calculate the number of images in training list
    num_img_inclass = size(find(label==label(rawidx)), 1);
 
    if num_positives > num_img_inclass
        new_num_negtives = num_positives + num_negtives - num_img_inclass;
        new_num_positives = num_img_inclass;
    else
        new_num_positives = num_positives;
        new_num_negtives = num_negtives;
    end
    % sort the distance vector and then get the corresponding sorted index
    % of images in raw file
    [~, Index] = sort(distmatrix(:, rawidx), 1, 'ascend');
    label_sorted = label(Index);

    % find closest/hard negtive samples
    neg_flag = new_num_negtives ;

    for idx = 1:num_img
        if neg_flag <= 0
            break;
        elseif label_sorted(idx) ~= label_sorted(1)
            neg_sample_idx(neg_flag, 1) = Index(idx, 1);
            neg_flag = neg_flag - 1;
        end
    end

    % find farthest positive samples
    pos_flag = new_num_positives;

    for idx = num_img:-1:1
        if pos_flag <= 0 
            break;
        elseif label_sorted(idx) == label_sorted(1)
            pos_sample_idx(pos_flag, 1) = Index(idx, 1);
            pos_flag = pos_flag - 1;
        end
    end

    % write in .txt file

    fprintf(gen_file, '%s %d\n', strcat(rootpath, path{rawidx}), label(rawidx));

    for idx = 1:new_num_positives
        header = path{pos_sample_idx(idx, 1)};
        pos_label = label(pos_sample_idx(idx, 1));
        fprintf(gen_file, '%s %d\n', strcat(rootpath, header), pos_label);
    end

    for idx = 1:new_num_negtives
        header = path{neg_sample_idx(idx, 1)};
        neg_label = label(neg_sample_idx(idx, 1));
        fprintf(gen_file, '%s %d\n', strcat(rootpath, header), neg_label);
    end
    disp(strcat('finished___', num2str(rawidx)));
end

fclose('all');
