function [grp_names,grp_means,grp_indices,n_grps,n_each_grp] = grpMean(group,data)
%Collapses columns in data matrix (num observations x num individuals) by group in group vector (1 x num species). 
    
    % INPUT
    % group: cell vector containing strings indicating group membership for each item (n items x 1)
    % data: 2D matrix of data to be collapsed (var x n items)
    
    %OUTPUT
    % grp_names: 
    % grp_means: 
    % grp_indices: 
    % n_grps: 
    % n_each_grp: 

    grp_names = unique(group(1,:),'stable'); %group names, in original order
    n_grps = length(grp_names); %number of groups
    n_each_grp = zeros(1,n_grps); %number of items for each group
    grp_means = zeros(size(data,1),n_grps); %mean for each group
    grp_indices = zeros(2,n_grps); %indices of first & last item for each group
    for i = 1:n_grps
        %sum images for each group
        for j = 1:size(data,2)
            if strcmp(grp_names{1,i},group{1,j})
                grp_means(:,i) = grp_means(:,i)+double(data(:,j));
                n_each_grp(1,i) = n_each_grp(1,i)+1;
                if grp_indices(1,i)==0
                    grp_indices(1,i) = j;
                    grp_indices(2,i) = j;
                else
                    grp_indices(2,i) = j;
                end
            end
        end
        
        %divide group sums by n groups
        grp_means(:,i) = grp_means(:,i)/n_each_grp(1,i);
    end
end

