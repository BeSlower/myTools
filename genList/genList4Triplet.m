clc;clear;
% load trainlist raw file
train = importdata('train.txt');
data = train.data;
path = train.textdata;

% new file which is going to write
filename = 'train_proc.txt';
% add root path into the front of image list path
rootpath = '/path/to/the/root/of/dataset/';

num_category = 4250;           % the number of categories
num_neg = 1;                   % the number of negtive samples
[num_img, ~] = size(data);     % the number of total images

glob_idx = 1;
loc_idx = 1;

% preprocess the raw data into proper format 
% (ascending order based on label and aggregate 
% the same label into one class struct)

for label = 1:num_category
    for idx_img = 1:num_img
        if data(idx_img,:) == (label-1)
            class{label}.globalId(loc_idx, 1) = glob_idx;
            class{label}.ordered_data(loc_idx, 1) = label;
            class{label}.ordered_path{loc_idx, 1} = path{idx_img};
            ordered_data(glob_idx, :) = label;
            ordered_path{glob_idx, 1} = path{idx_img};
            
            glob_idx = glob_idx + 1;
            loc_idx = loc_idx + 1;
        end
    end
    loc_idx = 1;
end



for idx_img = 1:num_img
    label = ordered_data(idx_img);
    % the label index and image index of anchor
    anchor_idx = [label, idx_img];
    
    % generate the label index and image index of positive sample
    [num_img_in_class, ~] = size(class{label}.ordered_data);
    in_class_idx = 1;
    while(class{label}.globalId(in_class_idx, 1) ~= idx_img)
        in_class_idx = randperm(num_img_in_class, 1);
        break;
    end
    pos_idx = [label, in_class_idx];
    
    % generate the label index and image index of negtive samples
    neg_label = label;
    while( size(find( neg_label == label), 2))
        neg_label = randperm(4250, num_neg);
    end
    for idx = 1:num_neg      
        [num_img_in_class, ~] = size(class{neg_label(idx)}.ordered_data);
        in_class_idx = randperm(num_img_in_class, 1);
        neg_idx{idx} = [neg_label(idx), in_class_idx];
    end
    
    % summarize the anchor, positive sample and negtive samples
    procData{idx_img}.anchor   = anchor_idx;
    procData{idx_img}.positive = pos_idx;
    for idx = 1:num_neg
       procData{idx_img}.negtive{idx} = neg_idx{idx}; 
    end
end

% generate the .txt file
gen_file = fopen(filename, 'wt');
rand_img_idx = randperm(num_img, num_img);

for idx = 1:num_img
	data_idx = rand_img_idx(idx);
    % anchor
    label = procData{data_idx}.anchor(1);
    header = ordered_path{data_idx};
    fprintf(gen_file, '%s %d\n', strcat(rootpath, header), label);
    
    % positive sample
    ps_label = procData{data_idx}.positive(1);
    ps_img_in_class_idx = procData{data_idx}.positive(2);
    ps_header = class{ps_label}.ordered_path{ps_img_in_class_idx};
    fprintf(gen_file, '%s %d\n', strcat(rootpath, ps_header), ps_label);
    
    % negtive samples
    for neg_idx = 1:num_neg
        ng_label = procData{data_idx}.negtive{neg_idx}(1);
        ng_img_in_class_idx = procData{data_idx}.negtive{neg_idx}(2);
        ng_header = class{ng_label}.ordered_path{ng_img_in_class_idx};
        fprintf(gen_file, '%s %d\n', strcat(rootpath, ng_header), ng_label);
    end
end
fclose('all');

