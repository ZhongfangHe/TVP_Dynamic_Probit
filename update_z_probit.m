% Update the latent index of static probit model:
% yt = I{zt > 0} or sign(zt),
% zt = xt' * beta + N(0,1),
% 
% p(zt|y,x,beta,z{-t}) \prop p(zt|x,beta) * p(yt|zt)

function z = update_z_probit(y, z, z_mean)
% Inputs:
%   y: a n-by-1 vector of binary target (1/0, or 1/-1),
%   x: a n-by-K matrix of regressors (first column is ones),
%   z: a n-by-1 latent index,
%   z_mean: a n-by-1 vector of the mean of z (i.e. xt'*beta),
% Outputs:
%   z: updated z


n = length(y);

% Draw latent index z
for t = 1:n
    zt_mean = z_mean(t);
    zt_std = 1;
    tmp = -zt_mean/zt_std;
    
    if y(t) > 0 %zt > 0
        zt = zt_mean + zt_std * trandn(tmp, Inf);
    else %y(t) = 0 or -1 <=> zt <= 0
        zt = zt_mean + zt_std * trandn(-Inf,tmp);
    end

    z(t) = zt;
end


