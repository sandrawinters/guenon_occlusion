function [] = occlusionAnalysis(ims,correct_label,filenames,block_radius,eigenfaces,features,labels,mean_face,image_dims,dist_metric,occlusion_color,resFoldAppend)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if length(occlusion_color)==1
    occlusion_color = repmat(occlusion_color,[1 3]);
end

for i=1:size(ims,2)
    test_im = ims(:,i);
  
    %classify image
    class = classifySpp(test_im,eigenfaces,features,labels,mean_face,image_dims,dist_metric,0); 
    correct = strcmp(class,correct_label);
    
    save(strcat('Classification_results',resFoldAppend,'/',filenames{i},'_classification.mat'),'class','correct')

    %run occlusion analysis on correctly classified images
    if correct==1
        test_im = reshape(test_im,image_dims);
        binary_im = ones(image_dims(1),image_dims(2));
        for j=block_radius+1:size(test_im,1)-block_radius
            for k=block_radius+1:size(test_im,2)-block_radius
                %occlude image
                occluded_im = test_im;
                occluded_im(j-block_radius:j+block_radius,k-block_radius:k+block_radius,1) = occlusion_color(1);
                occluded_im(j-block_radius:j+block_radius,k-block_radius:k+block_radius,2) = occlusion_color(2);
                occluded_im(j-block_radius:j+block_radius,k-block_radius:k+block_radius,3) = occlusion_color(3);
                
                %classify occluded image
                occluded_class = classifySpp(occluded_im(:),eigenfaces,features,labels,mean_face,image_dims,dist_metric,0); 

                %determine if classification is correct & record in binary image
                binary_im(j,k) = strcmp(occluded_class,correct_label); %make pixel black in binary image if occluded image is not correctly classified
            end
        end
        %save binary image
        imwrite(binary_im,strcat('Binary_images',resFoldAppend,'/',filenames{i},'_occlusion_results.tiff'));
    end
end

end

