function [eigenfaces,features,mean_face] = calcEigenfaces(ims,image_dims,n_eig,figures)
%Calculates n_eig eigenfaces & features from ims images
    
    % INPUT
    % ims: 2D matrix of vectorized images, n pixels x n images
    % n_eig: number of eigenfaces to return
    % image_dims: numeric vector indicating dimensions of images (when not vectorized)
    % figures: boolean value indicating whether to display figures (optional; defaults to false)
    
    % OUTPUT
    % eigenfaces: 2D matrix of vectorized eigenfaces, n pixels x n_eig
    % features: 2D matrix of feature weights for each image, n_eig x n images
    % mean_face: average face (vectorized), n pixels x 1
    
    %check args & set defaults
    if nargin==4
        %skip
    elseif nargin > 4 
        error('Error in calcEigenfaces: too many arguments')
    elseif nargin < 4
        figures = false;
    elseif nargin < 3 
        figures = false;
        n_eig = size(ims,2);
    else
        error('Error in calcEigenfaces: must include matrix of vectorized images & image dims as arguments');
    end
    
    %modify image_dims for grayscale images (for reshape)
    if size(image_dims,2)<3
        image_dims = [image_dims 1];
    end
    
    %set number of images
    n_im = size(ims,2);

    %find mean train image & mean-shifted train images
    mean_face = mean(ims,2);
    if figures==true 
        figure; imshow(reshape(mean_face,image_dims)); title('Mean training face')
    end

    train_shifted = ims - repmat(mean_face,1,n_im);
    if figures==true 
        figure; montage(reshape(train_shifted,[image_dims n_im])); title('Mean-shifted training faces')
    end
    
    %compute first n_eig principle components of cov matrix
    if n_eig==n_im
        [evectors,evalues] = eig(train_shifted.'*train_shifted);
    else
        [evectors,evalues] = eigs(train_shifted.'*train_shifted,n_eig);
    end
    
    s = sqrt(evalues);
    
    %calculate eigenfaces & features
    eigenfaces = fliplr(train_shifted*evectors*s^(-1));
    features = eigenfaces.'*train_shifted;
    
    if figures==true
        %display eigenfaces as montage
        eigenfaces_disp = uint8(((eigenfaces.*12)+(80/255)).*255);
        figure; montage(reshape(eigenfaces_disp,[image_dims n_eig])); title('Eigenfaces')
        
%         %display eigenfaces using subplot (spaces between them)
%          figure; title('Eigenfaces')
%          for cnt = 1:n_eig%n_eig:-1:1
%              t = reshape(eigenfaces(:,cnt), image_dims);
%              t = ((t.*12)+(80/255)).*255;
%              t = uint8(t);
%              subplot(5,6,cnt);
%              imshow(t)
%          end
    end
    
    eigenfaces = real(eigenfaces);
    features = real(features);
    
    %reconstruct faces
    if figures==true
        %display using montage
        recon = repmat(mean_face,[1,n_im]);
        for i = 1:n_eig
            recon = recon+(repmat(eigenfaces(:,i),[1 n_im]).*repmat(features(i,:),[size(recon,1) 1]));
        end
        figure; montage(reshape(recon,[image_dims n_im])); title('Reconstructed images')
        
%         %display using subplot
%         figure;
%         for i = 1:n_im
%              recon = mean_face;
%              for j = 1:n_eig
%                  recon=recon+(eigenfaces(:,j).*features(j,i));
%              end
%              temp_im = reshape(recon, image_dims);
%              subplot(5,7,i); imshow(temp_im)
%          end
     end
end
