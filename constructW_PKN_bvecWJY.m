% construct similarity matrix with probabilistic k-nearest neighbors. It is a parameter free, distance consistent similarity.
function W = constructW_PKN_bvecWJY(barY, Y, k, issymmetric)

if nargin < 4
    issymmetric = 1;
end;
if nargin < 3
    k = 5;
end;

[n1, n2, n3] = size(barY);
% % [dim, n] = size(X);
% % D = L2_distance_1(X, X);
% % [dumb, idx] = sort(D, 2); % sort each row
% find samples adjacency graph, KNN对数据分类
[KJL,D,~,~] = KJL_Tensorlateral(barY,6);%D对应diffMat
[dumb, idx] = sort(D, 2); % sort each row

W = zeros(n2);
%% ori
% for i = 1:n2
%     id = idx(i,2:k+2);%k+1个
%     di = D(i, id);%k+1个
%     W(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);%给第i个样本的近邻样本重新赋权
% end;

%% WJY
% beta=1;
     for i=1:1:n2
         for j=1:1:n2
%              distXzv(i,j)=norm(Xzv(:,i)-Xzv(:,j))^2;
             WN1(i,j)=exp(-(norm(Y(:,i,:) - Y(:,j,:),'fro'))/10);%取10 or 1

         end
     end
     
%  WN2=WN1;%其他情况都为相似度

     for i=1:1:n2 
        for j=1:1:k
            WN2(i,KJL(i,j))=WN1(i,KJL(i,j));%原来此处为0——条件太强硬
        end
     end
  
     for i=1:1:n2 
        for j=1:1:k
            WN2(KJL(i,j),i)=WN1(KJL(i,j),i);
        end
     end 
 %为了让WN保持对称   
 W=max(WN2,(WN2)'); 
end



% % if issymmetric == 1
% %     W = (W+W')/2;
% % end;
% 
% 
% 
% 
% % compute squared Euclidean distance
% % ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B
% function d = L2_distance_1(a,b)
% % a,b: two matrices. each column is a data
% % d:   distance matrix of a and b
% 
% 
% 
% if (size(a,1) == 1)
%   a = [a; zeros(1,size(a,2))]; 
%   b = [b; zeros(1,size(b,2))]; 
% end
% 
% aa=sum(a.*a); bb=sum(b.*b); ab=a'*b; 
% d = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab;
% 
% d = real(d);
% d = max(d,0);
% 
% % % force 0 on the diagonal? 
% % if (df==1)
% %   d = d.*(1-eye(size(d)));
% % end





