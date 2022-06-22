% Estimate probit models for recession data
% RHS shrinkage

clear;
rng(1234);
dbstop if warning;
dbstop if error;


%% Forecast horizon
for h = 4:4  %prediction horizon: zt = c * ztm1 + a * xtmh + ut
disp(['horizon h = ', num2str(h)]);


%% Model list
md = {'DTns','DTRHS','PTRHS'};
for mj = 1:3  
disp(['Model = ', md{mj}]);


%% Read data to get y, x
read_file = 'Data_Recession.xlsx';
read_sheet = 'Data'; 
data_y = readmatrix(read_file, 'Sheet', read_sheet, 'Range', 'B2:B273'); %1954Q3 to 2021Q3
data_x = readmatrix(read_file, 'Sheet', read_sheet, 'Range', 'C2:M273'); %1954Q3 to 2021Q3,11 non-constant regressors


data = [data_y  data_x];
[ntotal,ndata] = size(data);
y = data(h+1:ntotal, 1); %recession indicator
x = data(1:ntotal-h, 2:ndata); %lagged regressors


%% Configuration
npred = 0; %number of predictions >= 0
if npred > 0
    ind_pred = 1;
else
    ind_pred = 0;
end

ind_sparse = 0; %sparsify non-constant coefs by SAVS
disp(['sparse = ', num2str(ind_sparse)]);

if ismember(mj, [1 2]) %TVP dynamic probit
    if h > 1
        ind_sim = 1; %if simulate beta_tph in multi-horizon prediction
    end
end %legacy; shouldn't modify



%% Set the size of estimation/prediction sample
[n,nx] = size(x); %non-constant regressors
if npred > 0
    nest = n + 1 - h - npred; %number of estimation data
else
    nest = n;
end
disp(['nobs = ', num2str(n), ', nx (non-constant) = ', num2str(nx)]);
disp(['nest = ', num2str(nest), ', npred = ', num2str(npred)]);


%% MCMC
burnin = 2000; 
ndraws = 5000*4; 
disp(['burnin = ', num2str(burnin), ', ndraws = ', num2str(ndraws)]);

if npred == 0 %in-sample estimation only   
    % Prepare data
    yest = y;
    xest = [ones(n,1)  normalize_data(x)];
        
    % Estimate
    tic;
    switch mj
        case 1 %TDP_NS
            draws = Est_DTns_constARu_joint(yest, xest, burnin, ndraws, ind_sparse, ind_pred);
        case 2 %TDP
            draws = Est_DTRHS7_constARu(yest, xest, burnin, ndraws, ind_sparse, ind_pred);
        case 3 %TVP_Probit
            draws = Est_PTRHS6_constu(yest, xest, burnin, ndraws, ind_sparse, ind_pred);      
    end
    disp('Estimation is finished!');
    save(['Est_', md{mj},'_h', num2str(h),'.mat'],'draws');
    toc;
else %out-of-sample forecasts
    logpredlike = zeros(npred,2);     
    for predi = 1:npred       
        % Get the estimation sample data
        nesti = nest + predi - 1;
        yi = y(1:nesti,:);
        xi = x(1:nesti,:); 
        
        % Prepare data
        yest = yi;
        xest = [ones(nesti,1)  normalize_data(xi)];      
        
        % estimate the model
        switch mj
            case 1 %TDP_NS
                draws = Est_DTns_constARu_joint(yest, xest, burnin, ndraws, ind_sparse, ind_pred);
            case 2 %TDP
                draws = Est_DTRHS7_constARu(yest, xest, burnin, ndraws, ind_sparse, ind_pred);
            case 3 %TVP_Probit
                draws = Est_PTRHS6_constu(yest, xest, burnin, ndraws, ind_sparse, ind_pred);      
        end    
        
        % prediction
        xmean = mean(xi)';
        xstd = std(xi)';
  
        xtph = [ones(h,1)  (x(nesti+1:nesti+h,:) - repmat(xmean',h,1))...
            ./repmat(xstd',h,1)];           
        ytph = y(nesti+h);
        
        if ismember(mj, [1 2]) %TVP dynamic probit
            if h > 1
                [ytph_logpdf, ytph_logpdf_vec] = Pred_DT_MultiH_constARu(draws, xtph, ytph, h, ind_sim);
            else 
                [ytph_logpdf, ytph_logpdf_vec] = Pred_DT_constARu(draws, xtph', ytph);
            end
        else %TVP probit
            if h > 1
                [ytph_logpdf, ytph_logpdf_vec] = Pred_PT_MultiH_constu(draws, xtph(h,:)', ytph, h);
            else 
                [ytph_logpdf, ytph_logpdf_vec] = Pred_PT_constu(draws, xtph', ytph);
            end            
        end
        
        % store log likelihoods
        logpredlike(predi,1) = ytph_logpdf;
        disp(['Prediction ', num2str(predi), ' out of ', num2str(npred), ' is finished!']);
        toc;   
        disp(' ');
    end 
end
end
end
















