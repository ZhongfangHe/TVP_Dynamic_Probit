% Compute the bounds for the AA representation of probit
% xt * b + zt > 0 if yt = 1
% xt * b + zt <= 0 if yt = 0
% lb = max{-zt/xt} over t in the set {yt=0,xt<0}U{yt=1,xt>0}
% ub = min{-zt/xt} over t in the set {yt=0,xt>0}U{yt=1,xt<0}
%
% be careful of numerical issue when xt is near zero
% trim the subset of t whose corresponding xt is near zero

function [lb, ub] = probit_AA_bounds(y, x, z)
% Inputs:
%   y: a n-by-1 vector of binary target
%   x: a n-by-1 vector of the data for coef b
%   z: a n-by-1 vector of the intercept
% Outputs:
%   lb: a scalar of the lower bound for b
%   ub: a scalar of the upper bound for b

xt_min = 0;
xt_max = 0;


idx_lb = find(or(and(y==0,x<xt_min), and(y==1,x>xt_max)));
if isempty(idx_lb)
    lb = -Inf;
else
    lb = max(-z(idx_lb)./x(idx_lb));
end


idx_ub = find(or(and(y==0,x>xt_max), and(y==1,x<xt_min)));
if isempty(idx_ub)
    ub = Inf;
else
    ub = min(-z(idx_ub)./x(idx_ub));
end

if lb > ub
    if abs(lb-ub) < 0.001 %numerical trivia
        lb = ub;
    else
        error('lb > ub!');
    end
end



