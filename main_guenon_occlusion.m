% Master script for guenon occlusion project
% Runs occlude-reclassify analysis based on hemi-faces split down midline (with buffer from other side of face)

% Sandra Winters <sandra.winters@bristol.ac.uk>

% Citation: 
% Winters S, Allen WL, Higham JP. 2019. The structure of species discrimination signals 
% across a primate radiation. eLife. https://doi.org/10.7554/eLife.47428

%% set parameters
RESCALE = 1;
N_EIGENFACES = 15;
DIST_METRIC = 'euclidean';
FIGURES = false;
BLOCK_RADIUS = 30;
SIDE = 'L'; %'R'
TYPE = 'sppMean'; %'grey'

load 'average_colors.mat'
load 'guenon_spp_nums.mat'

disp(['Running occlusion analysis based on hemi-faces (' SIDE ' side); occluder = ' TYPE '; rescale = ' num2str(RESCALE) '; block radius = ' num2str(BLOCK_RADIUS)])

%% set up directories
if exist(['Binary_images_' TYPE SIDE],'dir')==0
    mkdir(['Binary_images_' TYPE SIDE])
end

if exist(['Classification_results_' TYPE SIDE],'dir')==0
    mkdir(['Classification_results_' TYPE SIDE])
end

if exist(['Individual_heatmaps_' TYPE SIDE],'dir')==0
    mkdir(['Individual_heatmaps_' TYPE SIDE])
end

if exist(['Species_heatmaps_' TYPE SIDE],'dir')==0
    mkdir(['Species_heatmaps_' TYPE SIDE])
end

%% import image database
disp('Importing images')

%get original images
[ims_orig,im_info,image_dims_orig] = getFaces('Guenon_images_frontal',RESCALE,1);
ims_orig = double(ims_orig)./255;
image_dims = [image_dims_orig(1) round(image_dims_orig(2)/2)+BLOCK_RADIUS image_dims_orig(3)];

%generate hemi-faces
ims_mirrored = zeros(image_dims(1)*((image_dims(2)-BLOCK_RADIUS)*2)*image_dims(3),size(ims_orig,2));
ims = zeros(prod(image_dims),size(ims_orig,2));
for i = 1:size(ims_orig,2)
    im = reshape(ims_orig(:,i),image_dims_orig);
    
    if ismember(SIDE,{'L','l'})
        hemi_im = im(:,1:round(image_dims_orig(2)/2)+BLOCK_RADIUS,:);
        im_mirrored = [im(:,1:round(image_dims_orig(2)/2),:),flipdim(im(:,1:round(image_dims_orig(2)/2),:),2)]; 
    elseif ismember(SIDE,{'R','r'})
        hemi_im = im(:,floor(image_dims_orig(2)/2)-BLOCK_RADIUS+1:end,:);
        im_mirrored = [im(:,floor(image_dims_orig(2)/2)+1:end,:),flipdim(im(:,floor(image_dims_orig(2)/2)+1:end,:),2)]; 
    else
        error('Unknown side')
    end
    
    ims(:,i) = hemi_im(:);
    ims_mirrored(:,i) = im_mirrored(:);
end

%display images
if FIGURES==true
    figure; montage(reshape(ims_orig,[image_dims_orig size(ims_orig,2)])); title('Original images')
    figure; montage(reshape(ims,[image_dims size(ims,2)])); title('Hemi-images')
    figure; montage(reshape(ims_mirrored,[image_dims(1) ((image_dims(2)-BLOCK_RADIUS)*2) image_dims(3),size(ims_orig,2)])); title('Mirrored images')
end

%calculate individual means
[ind_labels,ind_means,ind_indices,n_ind,n_each_ind] = grpMean(im_info(2,:),ims);
if FIGURES==true
    figure; montage(reshape(ind_means,[image_dims n_ind])); title('Individual means')
end

ind_spp = im_info(4,ind_indices(1,:));

%calculate species means
[spp_labels,spp_means,spp_indices,n_spp,n_each_spp] = grpMean(ind_spp,ind_means);
if FIGURES==true
    figure; montage(reshape(spp_means,[image_dims n_spp])); title('Species means')
end

%% classify individual mean images by species using leave-one-out procedure
disp('Running leave-one-out procedure')

%set vars
ind_class = cell(1,n_ind);
ind_correct = zeros(1,n_ind);

%start parallel pool using max number of detected processors
nCPU = feature('numcores');
p = parpool(nCPU);

%run occlusion
parfor i=1:n_ind
    %remove test individual from database
    test_ind_im = ind_means(:,i);
    train_ims = ind_means;
    train_ims(:,i) = [];
    
    test_ind_spp = ind_spp{i};
    train_spp = ind_spp;
    train_spp(:,i) = [];
    
    [train_spp_labels,train_spp_means,train_spp_indices,train_n_spp,~] = grpMean(ind_spp,ind_means);
    
    %calculate eigenfaces
    [eigenfaces,features,mean_face] = calcEigenfaces(train_spp_means,image_dims,N_EIGENFACES,FIGURES);

    %classify test image using nearest neighbor
    ind_class{i} = classifySpp(test_ind_im,eigenfaces,features,train_spp_labels,mean_face,image_dims,DIST_METRIC,FIGURES); 
    ind_correct(i) = strcmp(ind_class{i},test_ind_spp);
    
    %get occlusion color
    if strcmp(TYPE,'grey')
        occlusion_color = 80/255;
    else 
        idx = find(contains(guenon_spp_nums(:,4),test_ind_spp));
        occlusion_color = sp_av_rgb(idx,:)/255;
    end
    
    %run occlusion analysis on each image of individual
    occlusion_ims = ims(:,ind_indices(1,i):ind_indices(2,i));
    occlusion_filenames = strrep(im_info(6,ind_indices(1,i):ind_indices(2,i)),'.tiff','');
    occlusionAnalysis(occlusion_ims,test_ind_spp,occlusion_filenames,BLOCK_RADIUS,eigenfaces,features,train_spp_labels,mean_face,image_dims,DIST_METRIC,occlusion_color,['_' TYPE SIDE])

    disp(i)
end

disp('Leave-on-out procedure complete')

%% calculate proportion of individual faces correctly classified
prop_ind_correct = sum(ind_correct)/length(ind_correct)
% prop_im_correct = sum(im_correct)/length(im_correct);

save(['Classification_results_' TYPE SIDE '/individual_classifications.mat'],'ind_class','ind_correct','prop_ind_correct')

%% calculate proportion of images correctly classified
im_class = cell(1,size(ims,2));
im_correct = zeros(1,size(ims,2));
for i=1:size(ims,2)
    filename = strrep(im_info{6,i},'.tiff','_classification.mat');
    load(['Classification_results_' TYPE SIDE '/' filename])
    im_class(i) = class;
    im_correct(i) = correct;
end

prop_im_correct = sum(im_correct)/length(im_correct)

%% create individual level heatmaps from binary images
disp('Creating individual heatmaps')

ind_filenames = cell(1,n_ind);
for i=1:n_ind
    ind_heatmap = zeros(image_dims(1),image_dims(2));
    n = 0;
    for j=ind_indices(1,i):ind_indices(2,i)
        ind_filenames(i) = cellstr([im_info{4,j} '_' im_info{2,j}, '_occlusion_results.tiff']);
        if im_correct(j) %exist(['Binary_images/' strrep(im_info{6,j},'.tiff','') '_occlusion_results.tiff'],'file')==2
            tmp_im = im2double(imread(['Binary_images_' TYPE SIDE '/' strrep(im_info{6,j},'.tiff','') '_occlusion_results.tiff']));
            ind_heatmap = ind_heatmap+tmp_im;
            n = n+1;
        end
    end
    if n>0
        ind_heatmap = ind_heatmap/n;
        imwrite(ind_heatmap,['Individual_heatmaps_' TYPE SIDE '/' im_info{4,j} '_' im_info{2,j}, '_occlusion_results.tiff'],'tiff');
    end
end

%% create species level heatmaps from individual heatmaps & mirror for full face
disp('Creating species heatmaps')

parfor i=1:n_spp
    spp_heatmap = zeros(image_dims(1),image_dims(2));
    n = 0;
    for j=spp_indices(1,i):spp_indices(2,i)
        if exist(['Individual_heatmaps_' TYPE SIDE '/' ind_filenames{j}],'file')==2
            tmp_im = im2double(imread(['Individual_heatmaps_' TYPE SIDE '/' ind_filenames{j}]));
            spp_heatmap = spp_heatmap+tmp_im;
            n = n+1;
        end
    end
    spp_heatmap = spp_heatmap/n;
    
    spp_heatmap = spp_heatmap(:,1:image_dims(2)-BLOCK_RADIUS,:);
    spp_heatmap = [spp_heatmap,flipdim(spp_heatmap,2)];
    
    imwrite(spp_heatmap,['Species_heatmaps_' TYPE SIDE '/' spp_labels{i}, '_occlusion_results.tiff'],'tiff');
end

disp('Analysis complete')

delete(p)

%% save analysis
save(['occlusion_analysis_' TYPE SIDE '_' strrep(strrep(datestr(now),':','-'),' ','_')])

