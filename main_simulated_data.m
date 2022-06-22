% Simulate a probit model with TVP:
% zt = xt * bt + ut, ut ~ N(0,1)
% bt = btm1 + N(0,diag(v2)), b0 ~ N(0, w)
% yt = I{zt > 0}


clear;
rng(123456);

md = {'DPTVP'};
mj = 1;
disp(md{mj});

n = 300; %number of data points
K = 10; %number of regressors
nrep = 20; %number of simulation samples
z0 = 0; %initial value of zt
for dj = 1:nrep
    x = [ones(n,1)  randn(n,K-1)]; %regressors
    eta = randn(n,1); %innovations of zt
    
    beta_true = zeros(n,K); %first coef (intercept) is a constant, second coef is TVP, others zero
    beta_true(:,1) = -0.2;
    beta0_true = 0;
    vtrue = 0.1; 
    for t = 1:n
        if t > 1
            beta_true(t,2) = beta_true(t-1,2) + vtrue * randn;
        else
            beta_true(t,2) = beta0_true + vtrue * randn;
        end
    end
    xb = sum(x .* beta_true,2);

    phi = 0.8; %AR coef
    ztrue = zeros(n,1);
    p1true = zeros(n,1);
    y = zeros(n,1);
    for t = 1:n
        if t > 2
            zt_mean = phi * ztrue(t-1) + xb(t);
        else
            zt_mean = phi * z0 + xb(t);
        end
        zt = zt_mean + eta(t);
        ztrue(t) = zt;
        p1true(t) = normcdf(zt_mean); 
        y(t) = (zt > 0);    
    end
    disp(['rep ', num2str(dj),': mean(y) = ', num2str(mean(y))]);

    write_file = ['SimData_',md{mj},'.xlsx'];
    write_sheet = ['D',num2str(dj)];
    xstr = cell(1,K);
    betastr = cell(1,K);
    for j = 1:K
        xstr{j} = ['x',num2str(j)];
        betastr{j} = ['beta',num2str(j)];
    end
    title = [{'y'} xstr {'z','p1'} betastr];
    writecell(title, write_file, 'Sheet', write_sheet, 'Range', 'A1');
    writematrix([y x ztrue p1true  beta_true], write_file, 'Sheet', write_sheet, 'Range', 'A2');
end






