function [KJL,diffMat,S_Y,SY] = KJL_Tensorlateral(data,k)
%% 

[n1, n2, n3] = size(data);
SY=zeros(n2,n2);
S_Y=zeros(n2,k);
KJL=zeros(n2,k);
for i=1:1:n2
    for j=1:1:n2
        diffMat(i,j) = norm(data(:,i,:) - data(:,j,:),'fro');%N*N
    end    
        [~ ,SY(i,:)] = sort(diffMat(i,:));%返回升序排序后的矩阵，SY为其索引列 
        S_Y(i,:)=SY(i,2:k+1);
        
        for ki=1:k
            KJL(i,ki)=S_Y(i,ki);
        end
end
 
end
