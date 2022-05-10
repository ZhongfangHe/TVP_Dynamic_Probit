% Estimate a dynamic probit model with TVP, no shrinkage
%
% const intercept and AR coef for the latent variable
%
% Gibbs blocks: beta_star, rho, (z, u,beta0,v)
% PXDA is possible
% ASIS for z and/or beta is possible

function draws = Est_DTns_constARu(y, x, burnin, ndraws, ind_sparse, ind_pred)
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
df1 = 0.5; 
df2 = 0.5; 
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



%% Set up adaptive MH 
KK = 2*(K+1);
pstar = 0.25;
tmp_const = -norminv(0.5*pstar);
AMH_c = 1/(KK * pstar * (1-pstar)) + (1-1/KK)*0.5*sqrt(2*pi)*...
    exp(0.5*tmp_const*tmp_const)/tmp_const;
logrw_pv = 0;
logrw_start_pv = logrw_pv;
drawi_start_pv = 0; 
pv_mean = zeros(KK,1);
pv_cov = zeros(KK,KK); %hyper-para for beta0,v



%% MCMC
draws.beta = cell(K-1,1);
for j = 1:K-1
    draws.beta{j} = zeros(ndraws,n);
end %TVP 
draws.z = zeros(ndraws,n);
draws.p1 = zeros(ndraws,n);
draws.corr_eps = zeros(ndraws,1);

draws.alpha0 = zeros(ndraws,K+1); %[u, beta0, rho]
draws.beta0_d = zeros(ndraws,K+1); %hyper-para for beta0
draws.v = zeros(ndraws,K-1);
draws.v_d = zeros(ndraws,K+1); %hyper-para for v

draws.logrw_pv = zeros(ndraws,1); 
draws.count_pv = 0;
draws.count_rho = 0;
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
    
    
    % Draw rho in (z-SA, beta-AA): independent MH
    rho_old = rho;
    
    zg = z(2:n) - u - x(2:n,2:K)*beta0 - xbeta_star(2:n,:)*v;
    xg = z(1:n-1);
    prop_var_inv = xg' * xg;
    prop_var = 1 / prop_var_inv;
    prop_mean = prop_var * xg' * zg;
    rho_new = prop_mean + sqrt(prop_var) * randn;
    
    count_rho = 0;
    if abs(rho_new) < (1-1e-6) 
        z1_mean = u + x(1,2:K)*beta0 + (x(1,2:K).*beta_star(1,:))*v;
        tmp = (z(1) - z1_mean)^2;
        z1_var_old = 1 + rho_old*rho_old*sz;
        loglmq_old = -0.5*log(z1_var_old) - 0.5 * tmp / z1_var_old;
        z1_var_new = 1 + rho_new*rho_new*sz;
        loglmq_new = -0.5*log(z1_var_new) - 0.5 * tmp / z1_var_new;
        
        logp = loglmq_new - loglmq_old;
        if log(rand) < logp
            rho = rho_new;
            if drawi > burnin
                count_rho = 1;
            end
        end
    end     
    
    
    % Draw z in (z-SA, beta-AA): truncated normal, integrate out u,beta0
    z1_var = 1 + rho*rho*sz;
    z1_std = sqrt(z1_var);
    rho2 = rho * rho;
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
    
    xx = x;
    xx(1,:) = xx(1,:) / z1_std;
    xxtrans = xx';
    xx1 = xxtrans;
    xx1(:,1) = xx1(:,1)/z1_std;
    xx2 = rho*xxtrans(:,2:n);
    xtimesH = [xx1(:,1:n-1)-xx2  xx1(:,n)]; %xx' * H 
    
    omega = diag(1./[su; psil]) + xx' * xx; 
    omega_z = HH - xtimesH' * (omega \ xtimesH); %precision of z 
    
    mv = (x(:,2:K).*beta_star) * v;
    mv(1,:) = mv(1,:) / z1_std;
    mv1 = mv;
    mv1(1) = mv1(1)/z1_std;
    Hmv = mv1 - [rho*mv(2:n); 0]; %H' * mv
    xtimesmv = xx' * mv;
    
    F = sparse(diag(2 * y - 1));
    g = sparse(zeros(n,1));
    M = omega_z; %precision of z
    mu_r = zeros(n,1); %covector of z
    z_new = HMC_exact(F, g, M, mu_r, false, 2, z);
    z = z_new(:,2); %HMC
    if or(max(z(y<=0)) > 0, min(z(y>0)) < 0)
        error('zt <0 when yt > 0');
    end      
    
    
    
    % Draw hyper-parameters for v, beta0: integrate out u, beta0, v
    z1_std = sqrt(1+rho*rho*sz);
    yint = z - [0; rho*z(1:n-1)];
    xint = x_xbeta_star;
    yint(1) = yint(1)/z1_std;
    xint(1,:) = xint(1,:)/z1_std;
    xtimesy = xint' * yint;
    xtimesx = xint' * xint;
    [logtaul, logtaujl, logcl, logtau, logtauj, logcv, count_pv, pv_mean, pv_cov, ...
        logrw_pv, drawi_start_pv, logrw_start_pv] = regularized_HS_int3(xtimesy, xtimesx,...
        logtaul, logtaujl, logcl, logtau, logtauj, logcv, taul0, tau0, cva, cvb, ...
        drawi, burnin, pv_mean, pv_cov, logrw_pv, drawi_start_pv, logrw_start_pv,...
        AMH_c, pstar, su);     
    psi = exp(logcv - log(1 + exp(logcv)*exp(-logtau)*exp(-logtauj)));    
    psil = exp(logcl - log(1 + exp(logcl)*exp(-logtaul)*exp(-logtaujl)));     
    
    

    % Draw u, beta0, v in (z-SA, beta-AA): linear reg
    zg = z - [0; rho*z(1:n-1)];
    xg = x_xbeta_star;
    z1_std = sqrt(1+rho*rho*sz);
    zg(1) = zg(1)/z1_std;
    xg(1,:) = xg(1,:)/z1_std;
    
    prop_var_inv = diag(1./[su; psil; psi]) + xg' * xg;
    prop_var = prop_var_inv \ eye(2*K-1);
    prop_mean = prop_var * xg' * zg;
    ubeta0v = mvnrnd(prop_mean, prop_var)';
    u = ubeta0v(1); 
    beta0 = ubeta0v(2:K);
    v = ubeta0v(K+1:2*K-1);
    
    
    
%^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
%     % ASIS for PX-DA
%     % Draw z in (z-SA, beta-AA): truncated normal
%     z1_std = sqrt(1+rho*rho*sz);
%     H(1,1) = 1/z1_std;
%     for t = 2:n
%         H(t,t-1) = -rho;
%     end
%     xz = [x  x(:,2:K).*beta_star];
%     xz(1,:) = xz(1,:) / z1_std;
%   
%     F = sparse(diag(2 * y - 1));
%     g = sparse(zeros(n,1));
%     M = H' * H; %precision of z
%     mu_r = H' * xz * [u; beta0; v]; %covector of z
%     z_new = HMC_exact(F, g, M, mu_r, false, 2, z);
%     z = z_new(:,2); %HMC
%     if or(max(z(y<=0)) > 0, min(z(y>0)) < 0)
%         error('zt <0 when yt > 0');
%     end 
%     
%     
%     
%     % Draw working parameter
%     ig_a = 0.5*n;
%     
%     zz = [z(1)/z1_std; z(2:n)-rho*z(1:n-1)];
%     xx = [x  x(:,2:K).*beta_star];
%     xx(1,:) = xx(1,:)/z1_std;
%     omega = diag(1./[su; psil; psi]) + xx' * xx;
%     tmp = xx' * zz;
%     ig_b = 0.5 * (zz' * zz) - 0.5 * tmp' * (omega \ tmp);
%     
%     b2 = 1/gamrnd(ig_a, 1/ig_b);
%     b = sqrt(b2);
%     
%     
%     % Draw u, beta0, v in (z-SA, beta-AA): linear reg
%     zg = (z - [0; rho*z(1:n-1)]) / b;
%     xg = [x  x(:,2:K).*beta_star];
%     z1_std = sqrt(1+rho*rho*sz);
%     zg(1) = zg(1)/z1_std;
%     xg(1,:) = xg(1,:)/z1_std;
%     
%     prop_var_inv = diag(1./[su; psil; psi]) + xg' * xg;
%     prop_var = prop_var_inv \ eye(2*K-1);
%     prop_mean = prop_var * xg' * zg;
%     ubeta0v = mvnrnd(prop_mean, prop_var)';
%     u = ubeta0v(1); 
%     beta0 = ubeta0v(2:K);
%     v = ubeta0v(K+1:2*K-1);        
%^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%     % ASIS in z-AA: Compute eps, conditional on beta_star
%     a0v = [u; beta0; v];
%     x_zAA = [x  x(:,2:K).*beta_star];
%     eps = z - [0; rho*z(1:n-1)] - x_zAA*a0v;
% 
%     
%     % ASIS in z-AA: draw u,beta0,v
%     H(1,1) = 1;
%     for t = 2:n
%         H(t,t-1) = -rho;
%     end
%     ubeta0v = H \ eps; %intercept for a0v        
%     xbeta0v = H \ x_zAA; %slope matrix for a0v       
%     a0v_prior_mean = zeros(2*K-1,1);
%     a0v_prior_var = [su; psil; psi]; %prior of a0v
%     
% %     for jj = 1:50
% %         a0v = update_beta_probit_AA(y, xbeta0v, ubeta0v, a0v, ...
% %             a0v_prior_mean, a0v_prior_var); %Gibbs
% %     end %nested Gibbs
% 
% %     F = diag(2*y - 1) * xbeta0v;
% %     F = sparse(1:n,1:n,2*y-1) * xbeta0v;
%     F = repmat(2*y-1,1,2*K-1) .* xbeta0v;
%     g = (2*y-1) .* ubeta0v;
%     M = diag(a0v_prior_var); %covariance matrix
%     mu_r = a0v_prior_mean; %mean
%     a0v_new = HMC_exact(F, g, M, mu_r, true, 2, a0v);
%     a0v = a0v_new(:,2); %HMC  
%     
%     u = a0v(1);
%     beta0 = a0v(2:K);
%     v = a0v(K+1:2*K-1); 
%     
%      
%     % ASIS in z-AA: compute back z
%     z = xbeta0v * a0v + ubeta0v;
%     if or(max(z(y<=0)) > 0, min(z(y>0)) < 0)
%         error('zt <0 when yt > 0');
%     end         
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%**************************************************************************        
%     % ASIS in beta-SA: compute beta
%     beta = repmat(beta0',n,1) + beta_star .* repmat(v',n,1);
%     diff_beta = [beta(1,:)-beta0'; beta(2:n,:)-beta(1:(n-1),:)];
%     diff_beta2 = diff_beta.^2;  
%     
%     
%     % ASIS in beta-SA: re-draw v
%     vsign = sign(v);
%     v2 = v.^2;
%     for j = 1:K-1
%         vj2_old = v(j)^2;
%         aj = psi(j); %vj2~G(0.5,2*aj)
%         sum_diff_betaj = sum(diff_beta2(2:n,j)); 
%         vj2_new = gigrnd(1-0.5*n, 1/aj, sum_diff_betaj, 1);
%         
%         logprior_old = -0.5*log(vj2_old) - 0.5 * vj2_old / aj;
%         logprior_new = -0.5*log(vj2_new) - 0.5 * vj2_new / aj;
%         
%         vbeta0j = psil(j);
%         loglike_old = -0.5*log(vbeta0j+vj2_old) - 0.5*(beta(1,j)^2)/(vbeta0j+vj2_old) ...
%             -0.5*(n-1)*log(vj2_old) - 0.5 * sum_diff_betaj / vj2_old; 
%         loglike_new = -0.5*log(vbeta0j+vj2_new) - 0.5*(beta(1,j)^2)/(vbeta0j+vj2_new) ...
%             -0.5*(n-1)*log(vj2_new) - 0.5 * sum_diff_betaj / vj2_new; 
%         
% %         logprop_old = (1-0.5*n)*log(vj2_old) - 0.5*vj2_old/aj - 0.5*sum_diff_betaj/vj2_old;
% %         logprop_new = (1-0.5*n)*log(vj2_new) - 0.5*vj2_new/aj - 0.5*sum_diff_betaj/vj2_new;
%         logprop_old = -0.5*n*log(vj2_old) - 0.5*vj2_old/aj - 0.5*sum_diff_betaj/vj2_old;
%         logprop_new = -0.5*n*log(vj2_new) - 0.5*vj2_new/aj - 0.5*sum_diff_betaj/vj2_new;        
%         
%         logprob = (logprior_new + loglike_new - logprop_new) - ...
%             (logprior_old + loglike_old - logprop_old);
%         if log(rand) < logprob
%             v2(j) = vj2_new;
%         end     
%     end %integrate out beta0 when drawing v
%     v = sqrt(v2) .* vsign;
%     
% %     vsign = sign(v);
% %     v2 = v.^2;    
% %     for j = 1:K-1
% %         [v2(j),~] = gigrnd(0.5-0.5*n, 1/(tau*tauj(j)), sum(diff_beta2(:,j)), 1);
% %         if v2(j) == 0
% %             v2(j) = minNum;
% %         end 
% %     end
% %     v = sqrt(v2) .* vsign; %nested Gibbs for v, beta0
%     
%     
%     % ASIS in beta-SA: re-draw beta0
%     B_beta0 = 1./(1./psil + 1./v2);
%     b_beta0 = psil .* (beta(1,:)')./(v2 + psil);
%     beta0 = b_beta0 + sqrt(B_beta0).*randn(K-1,1);
%     
%     % ASIS in beta-SA: compute back beta_star
%     beta_star = (beta - repmat(beta0',n,1)) ./ repmat(v',n,1);
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
        draws.beta0_d(i,:) = [logcl  logtaul  logtaujl'];
        draws.v(i,:) = v';
        draws.v_d(i,:) = [logcv  logtau  logtauj'];
        
        draws.logrw_pv(i) = logrw_pv;
        draws.count_pv = draws.count_pv + count_pv/ndraws;      
        draws.count_rho = draws.count_rho + count_rho/ndraws;
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







