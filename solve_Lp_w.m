function   x   =  solve_Lp_w( y, lambda, p )

%这个代码就是输出阈值化后的奇异值,(补充:输入lambda对应原文Algorith1中的w)

% Modified by Dr. xie yuan
% lambda here presents the weights vector
J     =   4;  %2
% tau is generalized thresholding vector
tau   =  (2*lambda.*(1-p)).^(1/(2-p)) + p*lambda.*(2*(1-p)*lambda).^((p-1)/(2-p));%阈值
x     =   zeros( size(y) );
% i0表示 大于阈值tau的奇异值个数,
i0    =   find( abs(y)>tau );

if length(i0)>=1
    % lambda  =   lambda(i0);
    y0    =   y(i0);
    t     =   abs(y0);
    lambda0 = lambda(i0);
    for  j  =  1 : J
        t    =  abs(y0) - p*lambda0.*(t).^(p-1);%t对应S_p^(GST)
    end
    x(i0)   =  sign(y0).*t;%阈值化后的奇异值
end