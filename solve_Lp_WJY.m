function   x   =  solve_Lp_WJY( y, lambda, p )

% Modified by Dr. Weisheng Dong /Jingyu Wang M.*Z
J     =   2;
for i=1:1:size(lambda,1)
    for j=1:1:size(lambda,2)
        tau(i,j)   =  (2*lambda(i,j).*(1-p))^(1/(2-p)) + p*lambda(i,j).*(2*(1-p)*lambda(i,j)+eps)^((p-1)/(2-p));%matrix
    end
end
x     =   zeros( size(y) );
i0    =   find( abs(y) > tau );

if length(i0) >= 1
    % lambda  =   lambda(i0);
    y0    =   y(i0);
    t     =   abs(y0);
    for  j  =  1 : J
        t    =  abs(y0) - p*lambda(i0).*(t).^(p-1);%matrix
    end
    x(i0)   =  sign(y0).*t;
end
