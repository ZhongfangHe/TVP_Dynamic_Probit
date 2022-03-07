% Given y1, y2, ..., y(K+L-1), assemble a K-by-L matrix
% [yL       yLm1     ...  y1;
%  yLp1     yL       ...  y2;
%                    ...
%  y(K+L-1) y(K+L-2) ...  yK];

function ymat = AR_X_mat(y, L)
nn = length(y);
K = nn + 1 - L;
ymat = zeros(K,L);
for j = 1:L
    ymat(:,j) = y(L-j+1:L-j+K);
end

