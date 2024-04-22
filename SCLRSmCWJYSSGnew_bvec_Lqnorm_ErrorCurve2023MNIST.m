function [Z,iter,Error_ZC,Error_ZQ,Error_ZT,Error_SSG,Error_JTK] = SCLRSmCWJYSSGnew_bvec_Lqnorm_ErrorCurve2023MNIST(Y, M, lambda1, lambda2, mu, rho, mu2, lambda3)


% INPUT:
% Y: sample imaging data, every lateral slice in Y corresponds to one sample
% M: dissimilarity matrix calculated from the data
% lambda1, lambda2: regularization parameters
%
% OUTPUT:
% Z: coefficient tensor
%--------------------------------------------------------------------------
% Copyright @ Tong Wu and Waheed U. Bajwa, 2021
%--------------------------------------------------------------------------

[n1, n2, n3] = size(Y);
Yhat = fft(Y,[],3);

Z = zeros(n2,n2,n3);
C = zeros(n2,n2,n3);
Q = zeros(n2,n2,n3);

G1 = zeros(n2,n2,n3);
G2 = zeros(n2,n2,n3);
G3 = zeros(n2,n2,n3);
T = zeros(n2,n2,n3);
oldZ = Z; oldC = C; oldQ = Q; oldT = T; 


mu_max = 1e10;
epsilon = 1e-5;
maxIter = 100;
iter = 0;
p=0.5;

ww=ones(n3,1);

%% new add
%% normalize each lateral slice of tensor Y
for i=1:1:n2
    for j=1:1:n1
        for k=1:1:n3
            barY(j,i,k)=Y(j,i,k)/norm(Y(:,i,:),'fro');
        end
    end
end

    Weightk = constructW_PKN_bvecWJY(barY,Y, 5); 
    
    AK=Weight_vector(Weightk,n2);
    AKlength=size(AK,1);
    maxlength=max(AKlength);

    %AKnew
    for i=1:n3
        AK_new{i}=AK;
    end
    AK_tensor = cat(3, AK_new{:,:});

    for k=1:n3
        B1(:,:,k) = zeros(n2,maxlength);
        J(:,:,k) = zeros(n2,maxlength);
    end
    oldJ = J;

while iter<maxIter
    iter = iter + 1;
    
    % update C(w-Sp)
    A = Z + G1/mu;

    [C,~,~] = prox_tnn_shiftdim(A,ww/mu,p,3);
    

    
    % update Q(Lp-norm)
    Qtemp = Z + G2/mu;
    
    for i = 1:n3
        Q(:,:,i) =  solve_Lp_WJY( Qtemp(:,:,i), (lambda1/mu).*M, 0.5 );
    end
    
    % update Z
    P1 = C - G1/mu;
    P2 = Q - G2/mu;
    P3 = T - G3/mu;
    
    P1hat = fft(P1,[],3);
    P2hat = fft(P2,[],3);
    P3hat = fft(P3,[],3);    
    Zhat = zeros(size(C));
    
    for i = 1:n3
        Zhat(:,:,i) = (2*lambda2*Yhat(:,:,i)'*Yhat(:,:,i) + 3*mu*eye(n2))\(2*lambda2*Yhat(:,:,i)'*Yhat(:,:,i) + mu*(P1hat(:,:,i) + P2hat(:,:,i) + P3hat(:,:,i)));
    end    
    Z = ifft(Zhat,[],3);
    %% add 对角线元素为0

    for i = 1:n3
        Z(:,:,i)=Z(:,:,i)-diag(diag(Z(:,:,i)));
    end


%%
    % update T(Z's auxiliary variable)
    for i = 1:n3
        T(:,:,i) = (mu2*J(:,:,i)*AK_tensor(:,:,i)-B1(:,:,i)*AK_tensor(:,:,i)+mu*Z(:,:,i)+G3(:,:,i))*inv((mu2*AK_tensor(:,:,i)'*AK_tensor(:,:,i)+mu*eye(n2)));
    end
    
    % update J^k
    for k=1:n3
%         J(:,:,k)=errormin(B1(:,:,k),AK_tensor(:,:,k),T(:,:,k),lambda3,mu2,1);
        tempJ(:,:,k)=T(:,:,k)*AK_tensor(:,:,k)'+B1(:,:,k)/mu2;
        J(:,:,k)=max(abs(tempJ(:,:,k))-(lambda3/mu2)*ones(n2,maxlength),0).*sign(tempJ(:,:,k));%WJY
    end
    
 
%% new add
        for k=1:1:n3
            Error_JTK(1,k)=max(max(T(:,:,k)*AK_tensor(:,:,k)'-J(:,:,k)));% 向量
        end
    stopC = max([ max(max(max(abs(Z - C)))), max(max(max(abs(Z - Q)))),max(max(max(abs(Z - T)))), max(max(max(abs(Z - oldZ)))), max(max(max(abs(C - oldC)))), max(max(max(abs(Q - oldQ)))), max(max(max(abs(T - oldT)))), max(max(max(abs(J - oldJ)))), max(Error_JTK)] );
    
    if stopC<epsilon
        break;
    else
        G1 = G1 + mu*(Z - C);
        G2 = G2 + mu*(Z - Q);
        G3 = G3 + mu*(Z - T);
        for k=1:1:n3
            B1(:,:,k)=B1(:,:,k)+mu2*(T(:,:,k)*AK_tensor(:,:,k)'-J(:,:,k));
        end
        mu = min(rho*mu, mu_max);
        mu2 = min(rho*mu2, mu_max);
        oldZ = Z; oldC = C; oldQ = Q; oldT = T; oldJ = J;
    end

Error_ZC(1,iter)=[max(max(max(abs(Z - C))))];    
Error_ZQ(1,iter)=[max(max(max(abs(Z - Q))))]; 
Error_ZT(1,iter)=[max(max(max(abs(Z - T))))]; 
Error_SSG(1,iter)=[max(Error_JTK)]; 

end
end