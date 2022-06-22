% Estimate a dynamic probit model with TVP, no shrinkage
%
% const intercept and AR coef for the latent variable
%
% Gibbs blocks: beta_star, z, (u,beta0,v,rho)
% ASIS for beta
%
% Inverse gamma prior for v2

function draws = Est_DTns_constARu_joint(y, x, burnin, ndraws, ind_sparse, ind_pred)
% Inputs:
%   y: a n-by-1 vector of binary target (1/0, or 1/-1),
%   x: a n-by-K matrix of regressors (first column is ones),
%   burnin: a scalar of the number of burn-ins,
%   ndraws: a scalar of the number of draws after burn-in.
%   ind_sparse: an indicator if sparsification for non-constant coefs is needed (not updated, should be 0)
%   ind_pred: an indicator if prediction is needed (run a Kalman filter)
% Outputs:
%   draws: a structure of the draws.

[n,K] = size(x);
minNum = 1e-100;
maxNum = 1e100;


%% Priors: 
% intercept, u ~ N(0,su)
su = 10;
u = sqrt(su) * randn;

% AR coefficient, rho ~ U(-1,1)
rho = rand;

% initial z, z0 ~ N(0,sz): will be integrated out
sz = 10;

% initial beta, beta0 ~ N(0, psil)
psil = 10 * ones(K-1,1); 
beta0 = sqrt(psil) .* randn(K-1,1);


% process SD, v2 ~ IG(a,b) => p(v) \prop v^{-2*a-1} exp(-b/v2) 
v2_mean = 0.5;
df1 = 1.2; % should be > 1 to have a valid mean 
df2 = v2_mean * (df1 - 1); 
% df1 = 0.5;
% df2 = 0.5;
v2 = 1./gamrnd(df1,1/df2,K-1,1);
v = sqrt(v2);



%% Initialize
beta_star = cumsum(randn(n,K-1));

z = zeros(n,1);
for t = 1:n
    tmp = u + x(t,2:K) * beta0 + sum(x(t,2:K)' .* v .* beta_star(t,:)'); 
    if t == 1
        zt_mean = tmp;
        zt_std = sqrt(1+rho*rho*sz); %integrate out z0
    else
        zt_mean = tmp + rho*z(t-1);
        zt_std = 1;
    end

    if y(t) > 0
        z(t) = zt_mean + zt_std * trandn(-zt_mean/zt_std,Inf);
    else
        z(t) = zt_mean + zt_std * trandn(-Inf,-zt_mean/zt_std);
    end
end


state_var = cell(n,1);
for t = 1:n
    state_var{t} = eye(K-1);
end
varz = ones(n,1); %variance of epsilon
H = eye(n); % AR slope matrix for z
HH = eye(n); % H'*H
F = sparse(diag(2 * y - 1)); %slope: constraints of z
gz = sparse(zeros(n,1)); %intercept: constraints of z


LL = 2*eye(n);
LL(n,n) = 1;
for t = 2:n
    LL(t,t-1) = -1;
    LL(t-1,t) = -1;
end
os = ones(n,1);
betaj_covector = zeros(n,1);
idxy0 = find(y==0);



%% MCMC
draws.beta = cell(K-1,1);
for j = 1:K-1
    draws.beta{j} = zeros(ndraws,n);
end %TVP 
draws.z = zeros(ndraws,n);
draws.p1 = zeros(ndraws,n);
draws.corr_eps = zeros(ndraws,1);

draws.alpha0 = zeros(ndraws,K+1); %[u, beta0, rho]
draws.v = zeros(ndraws,K-1);
 
draws.count_g = 0;
% draws.b = zeros(ndraws,1); %working para for PXDA

if ind_pred == 1
    draws.bn_mean = zeros(ndraws,K-1);
    draws.bn_cov = cell(ndraws,1);
    for j = 1:ndraws
        draws.bn_cov{j} = zeros(K-1,K-1);
    end
end

if ind_sparse == 1
    draws.v_sparse = zeros(ndraws,K+1);
    draws.beta0_nonconst_sparse = zeros(ndraws,K);
    draws.beta_sparse = cell(K+1,1);
    for j = 1:K+1
        draws.beta_sparse{j} = zeros(ndraws,n);
    end
    draws.p1_sparse = zeros(ndraws,n);
end

ntotal = burnin + ndraws;
tic;
for drawi = 1:ntotal   
    % Draw beta_star in (z-SA, beta-AA): simulation smoother
    zstar = z - u - [0; rho*z(1:n-1)] - x(:,2:K) * beta0;
    xstar = x(:,2:K) .* repmat(v',n,1);
    varz(1) = 1 + rho*rho*sz;
    beta_star = Simulation_Smoother_DK(zstar, xstar, varz, state_var(2:n),...
        zeros(K-1,1), state_var{1});
    xbeta_star = x(:,2:K).*beta_star;
    x_xbeta_star = [x  xbeta_star];  
    
    
    % Draw z in (z-SA, beta-AA): truncated normal, conditional on u,beta0,v
    rho2 = rho * rho;
    z1_var = varz(1);
    z1_std = sqrt(z1_var);    
    for t = 1:n
        if t > 1
            HH(t,t-1) = -rho;
        end
        if t < n
            HH(t,t+1) = -rho;
        end
        if t == 1
            HH(t,t) = 1/z1_var + rho2;
        elseif t < n
            HH(t,t) = 1 + rho2;
        end
    end %tridiagonal matrix H'*H
    
    xx = x_xbeta_star;
    xx(1,:) = xx(1,:) / z1_std;
    xxtrans = xx';
    xx1 = xxtrans;
    xx1(:,1) = xx1(:,1)/z1_std;
    xx2 = rho*xxtrans(:,2:n);
    xtimesH = [xx1(:,1:n-1)-xx2  xx1(:,n)]; %xx' * H 
    
    M = HH; %precision of z
    mu_r = xtimesH' * [u; beta0; v]; %covector of z
    z_new = HMC_exact(F, gz, M, mu_r, false, 2, z);
    z = z_new(:,2); %HMC
    if or(max(z(y<=0)) > 0, min(z(y>0)) < 0)
        error('zt <0 when yt > 0');
    end  
    
    
    
    
    % Draw u,beta0,v,rho in (z-SA, beta-AA): MH
    g = [u; beta0; v; rho];
    g_old = g;
    u_old = u;
    beta0_old = beta0;
    v_old = v;
    rho_old = rho;
    
    
    x2n = [x_xbeta_star(2:n,:)  z(1:n-1)];
    z2n = z(2:n);
    prop_precision = x2n' * x2n;
    prop_covector = x2n' * z2n;
    tmp = mvnrnd(prop_covector, prop_precision)';
    g_new = prop_precision \ tmp; 
    u_new = g_new(1);
    beta0_new = g_new(2:K);
    v_new = g_new(K+1:2*K-1);
    rho_new = g_new(2*K);
    
    if abs(rho_new) < (1-1e-8)         
        v2_old = v_old.^2;
        v2_new = v_new.^2;
        logprior_old = -0.5*u_old*u_old/su - 0.5*sum(beta0_old.*beta0_old./psil) ...
            -0.5 * (1+2*df1) * sum(log(v2_old)) - df2 * sum(1./v2_old);
        logprior_new = -0.5*u_new*u_new/su - 0.5*sum(beta0_new.*beta0_new./psil) ...
            -0.5 * (1+2*df1) * sum(log(v2_new)) - df2 * sum(1./v2_new);


        tmp = z(1) - x_xbeta_star(1,:)*g_old(1:2*K-1);
        z1_var_old = 1 + rho_old*rho_old*sz; 
        loglike_old = -0.5*log(z1_var_old) - 0.5*tmp*tmp/z1_var_old;
        tmp = z(1) - x_xbeta_star(1,:)*g_new(1:2*K-1);
        z1_var_new = 1 + rho_new*rho_new*sz; 
        loglike_new = -0.5*log(z1_var_new) - 0.5*tmp*tmp/z1_var_new;    


        logprob = logprior_new + loglike_new - logprior_old - loglike_old;
        count_g= 0;
        if log(rand) <= logprob
            u = u_new;
            beta0 = beta0_new;
            v = v_new;
            rho = rho_new;
            if drawi > burnin
                count_g = 1;
            end
        end
    end
    z1_var = 1+rho*rho*sz;
    z1_std = sqrt(z1_var);
    


%**************************************************************************        
    % ASIS in beta-SA: compute beta
    beta = repmat(beta0',n,1) + beta_star .* repmat(v',n,1);
    diff_beta = [beta(1,:)-beta0'; beta(2:n,:)-beta(1:(n-1),:)];
    diff_beta2 = diff_beta.^2;  
    
    
    % ASIS in beta-SA: re-draw v    
%     vsign = sign(v);
    vsign = double(rand(K-1,1)<0.5)*2-1;
    v2 = v.^2;    
    for j = 1:K-1
        v2(j) = 1/gamrnd(df1+0.5*n, 1/(df2+0.5*sum(diff_beta2(:,j))));
    end
    v = sqrt(v2) .* vsign; %nested Gibbs for v, beta0
    
    
    % ASIS in beta-SA: re-draw beta0
    B_beta0 = 1./(1./psil + 1./v2);
    b_beta0 = psil .* (beta(1,:)')./(v2 + psil);
    beta0 = b_beta0 + sqrt(B_beta0).*randn(K-1,1);
    
    % ASIS in beta-SA: compute back beta_star
    beta_star = (beta - repmat(beta0',n,1)) ./ repmat(v',n,1);
%**************************************************************************
            
    
    % Compute probability
    beta = repmat(beta0',n,1) + beta_star .* repmat(v',n,1);
    z_mean = u + [0; rho*z(1:n-1)] + sum(x(:,2:K) .* beta,2);
    p1 = normcdf([z_mean(1)/sqrt(1+rho*rho*sz); z_mean(2:n)]);
    

    % Compute the first-order autocorrelation of latent index resid
    eps = z - z_mean;
    corr_eps = corr(eps(1:n-1),eps(2:n));    
    
    
    % Sparsify if needed
    if ind_sparse == 1
        xAR = x;
        xbeta_star = xAR .* beta_star;
        v_sparse = SAVS_vector(v, xbeta_star); 
        
        beta0_nonconst_sparse = SAVS_vector(beta0(2:K+1), xAR(:,2:K+1));
        beta0_sparse = [beta0(1); beta0_nonconst_sparse];
        
        beta_sparse = beta_star .* repmat(v_sparse',n,1) + repmat(beta0_sparse',n,1);         
    end 
   
    
    % Run Kalman filter if prediction is needed
    if ind_pred == 1
        zz = z - u - [0; rho*z(1:n-1)];
        xx = x(:,2:K);
        z1_std = sqrt(1+rho*rho*sz);
        zz(1) = zz(1)/z1_std;
        xx(1,:) = xx(1,:)/z1_std;       
        [bn_mean, bn_cov] = TVP_beta_filter(zz, xx, ...
            diag(v.^2), beta0);
    end
    

    % Collect draws
    if drawi > burnin
        i = drawi - burnin;
        for j = 1:K-1
            draws.beta{j}(i,:) = beta(:,j)';
        end
        draws.z(i,:) = z';
        draws.p1(i,:) = p1';
        draws.corr_eps(i) = corr_eps;

        draws.alpha0(i,:) = [u  beta0'  rho];
        draws.v(i,:) = v';
        
        draws.count_g = draws.count_g + count_g/ndraws;      
%         draws.b(i) = b;

        if ind_sparse == 1
            draws.v_sparse(i,:) = v_sparse';
            draws.beta0_nonconst_sparse(i,:) = beta0_nonconst_sparse';
            for j = 1:K+1
                draws.beta_sparse{j}(i,:) = beta_sparse(:,j)';
            end
            draws.p1_sparse(i,:) = normcdf(sum(xAR.*beta_sparse,2));
        end 
        
        if ind_pred == 1
            draws.bn_mean(i,:) = bn_mean';
            draws.bn_cov{i} = bn_cov;
        end          
    end
    
    
    % Display elapsed time
    if (drawi/5000) == round(drawi/5000)
        disp([num2str(drawi), ' out of ', num2str(ntotal),' draws have completed!']);
        toc;
    end
end







