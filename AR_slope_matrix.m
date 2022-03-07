% Given a vector a1,a2,...,an, return the n-by-n matrix:
% [an  anm1  anm2  ... a2  a1;
%  0   an    anm1  ... a3  a2; 
%  0   0     an    ... a4  a3; 
%  ...;
%  0   0     0     ... an  anm1; 
%  0   0     0     ... 0   an];
function a_matrix = AR_slope_matrix(a_vec)
n = length(a_vec);
a_matrix = zeros(n,n);
a_vec_ud = flipud(a_vec)';
for j = 1:n
    a_matrix(j,j:n) = a_vec_ud(1:n-j+1);
end

