function [images,info,image_dims] = getFaces(directory,resize,color)
%Returns matrix of images & associated info for guenon faces in directory, resized based 
%on resize parameter and converted to grayscale when color equals false. 

    %INPUT
    %directory: directory of images, asks for directory if none provided
    %resize: image resize parameter (for imresize), default resize=1 for no resizing
    %color: denotes color (true) or black and white (false) images, default color=true

    %OUTPUT
    %images: matrix of vectorized images (n pixels x n images)
    %info: matrix of image data: species #, individual ID, location, species name, individual image number
    %image_dims: image dimensions
    
    %set default parameters
    if nargin<3
        color = true;
    end
    
    if nargin<2 
        resize = 1;
    end
    
    if nargin==0 
        directory=uigetdir(cd,'Select directory of images');
    end
        
    %import images from directory
    filenames = dir(fullfile(directory,'*.tiff'));
    n_images = numel(filenames);
    info = cell(6,n_images);
    
    %create image & image info matrices
    for i = 1:n_images
        filename = fullfile(directory,filenames(i).name);
        
        %add image to matrix
        image = imread(filename);
        
        if color==false
            image = rgb2gray(image);
        end
        
        image = imresize(image, resize);
        images(:,i) = image(:);
        
        %store size of first image for future processing (assume all images are the same size)
        if i == 1
            image_dims = size(image);
        end

        %store image info
        tmp = strrep(filename,'\','/');
        tmp = strsplit(tmp,'/');
        filename = tmp(length(tmp));
        im_info = strsplit(filename{:},'_');
        info(1,i) = im_info(1,2); %species number
        info(2,i) = im_info(1,3); %individual ID
        info(3,i) = im_info(1,5); %location
        info(4,i) = cellstr(strcat(im_info{1,6},'_',im_info{1,7})); %species name
        info(5,i) = im_info(1,4); %individual image number
        info(6,i) = filename; %image name
    end
end

