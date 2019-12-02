function [classifications] = classifySpp(ims,eigenfaces,features,labels,mean_face,image_dims,dist_metric,figures)
%Classifies an image based on its nearest neighbor in feature space

%INPUT
    %ims: 2D matrix of vectorized images to be classified, n pixels x n images (e.g. from getFaces.m)
    %eigenfaces: 2D matrix of vectorized eigenfaces calculated from training images, n pixels x n_eig (e.g. from calcEigenfaces.m)
    %features: 2D matrix of feature weights for each training image, n_eig x n images (e.g. from calcEigenfaces.m)
    %labels: vector of labels for training images
    %mean_face: average face (vectorized) from training images, n pixels x 1 (e.g. from calcEigenfaces.m)
    %image_dims: numeric vector indicating dimensions of images (when not vectorized) (e.g. from getFaces.m)
    %dist_metric: method for calculating nearest neighbor, string passed to pdist function (e.g. 'euclidean')
    %figures: boolean value indicating whether to display figures (optional; defaults to false)

%OUTPUT
    %classifications: vector of classifications

%check args & set defaults
%TODO

%set n test images
n_ims = size(ims,2);

%calculate mean shifted test images
ims_shifted = ims - repmat(mean_face,1,n_ims); 
if figures==true
    figure; montage(reshape(ims_shifted,[image_dims n_ims])); title('Centered test faces');
end

%rotate eigenface & feature matrices for subsequent analyses
new_features = rot90(features);
new_labels = rot90(labels);

%classify new images
classifications = cell(1,n_ims); %repmat(char(0),[1,n_im]);
for i = 1:n_ims
    %project test images into subspace
    im_features = real(eigenfaces.'*ims_shifted(:,i));
    
    %calculate distances to species features
    all_features = [im_features';new_features];
    similarity_scores = squareform(pdist(all_features,dist_metric));
    similarity_scores = similarity_scores(1,2:end);

    %get minimum distance
    [~,nn] = min(similarity_scores);
    classifications(i) = new_labels(nn);
end

end

