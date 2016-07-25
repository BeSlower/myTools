clear; clc;
 
% initial setting
num_img = 42468;
num_positives = 64;
num_negtives = 64;

% indicate whether use random sample selection or not
random_flag = 1;  

if ~random_flag
    % load feature matrix
    load('feature.mat');
    
    % calculate the distance matrix
    disp('Starting distance matrix calculation')
    distmatrix = pdist2(feature, feature, 'cosine');
    disp('distance matrix calculation finished')
end

% write into .txt file setting
filename = 'train_batch128.txt';
rootpath = '/path/to/the/root/of/dataset/';
gen_file = fopen(filename, 'wt');

% pre-processing raw image list .txt file in order to
% generate labels(ground-truth) corresponding the index
train = importdata('train.txt');
label = train.data;
path = train.textdata;

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

    if random_flag
        % randomly select negtive samples
        for idx_for_neg = 1:new_num_negtives
            neg_sample_idx_ = randperm(num_img, 1);
            % if the label of neg_sample_idx_ image is same with the
            % label of raw image, re-sample
            while(label(neg_sample_idx_) == label(rawidx))
                neg_sample_idx_ = randperm(num_img, 1);
            end
            neg_sample_idx(idx_for_neg, 1) = neg_sample_idx_;
        end
        
        % randomly select positive samples
        for idx_for_pos = 1:new_num_positives
            pos_sample_idx_ = randperm(num_img, 1);
            % if the label of pos_sample_idx_ image isn't same with the
            % label of raw image OR the index of selected image is equal to
            % the index of raw image which means that the sampled image is
            % a same image of raw, re-sample
            while(label(pos_sample_idx_) ~= label(rawidx) || pos_sample_idx_ == rawidx)
                pos_sample_idx_ = randperm(num_img, 1);
            end
            pos_sample_idx(idx_for_pos, 1) = pos_sample_idx_;
        end
    else
        % sort the distance vector and then get the corresponding sorted index
        % of images in raw file
        [~, Index] = sort(distmatrix(:, rawidx), 1, 'ascend');
        label_sorted = label(Index);
        
        % find closest/hard negtive samples
        neg_flag = new_num_negtives;

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
