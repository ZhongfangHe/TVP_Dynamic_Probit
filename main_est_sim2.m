% Estimate probit model for simulated data


clear;
rng(1234567);
dbstop if warning;
dbstop if error;


ms = {'m10'};
nd = length(ms);

ind_sparse = 0;
ind_pred = 0;
burnin = 1000*2;
ndraws = 5000*4;
disp(['burnin = ', num2str(burnin), ', ndraws = ', num2str(ndraws)]);
 
write_file = 'RMSE_P1.xlsx';
write_col = {'A','B','C','D','E','F','G','H','I','J',...
             'K','L','M','N','O','P','Q','R','S','T'};
for repj = 1:1 %1:20
    tic;
    for dj = 1:nd    
        % read data to get y, x
        read_file = 'SimData_DPTVP.xlsx';
        read_sheet = ['D',num2str(repj)];
        data = readmatrix(read_file, 'Sheet', read_sheet, 'Range', 'A2:W301');
        y = data(:,1);
        x = data(:,2:11);
        btrue = data(:,14:23);
        ztrue = data(:,12);
        p1true = data(:,13);


        
        % estimate the TDP model
        draws = Est_DTRHS7_constARu(y, x, burnin, ndraws, ind_sparse, ind_pred);
        tmp = draws.p1 - repmat(p1true',ndraws,1); 
        rmse_p1 = sqrt(mean(tmp.^2))'; 
        write_sheet = [ms{dj},'_dths'];
        writematrix(rmse_p1, write_file, 'Sheet', write_sheet, 'Range', [write_col{repj},'2']);
        if repj == 1
            save(['Est_DTRHS_',ms{dj},'.mat'],'draws');
        end
        
        
        % estimate the TDP-NS model
        draws = Est_DTns_constARu_joint(y, x, burnin, ndraws, ind_sparse, ind_pred);
        tmp = draws.p1 - repmat(p1true',ndraws,1); 
        rmse_p1 = sqrt(mean(tmp.^2))';
        write_sheet = [ms{dj},'_dns'];
        writematrix(rmse_p1, write_file, 'Sheet', write_sheet, 'Range', [write_col{repj},'2']);         
        if repj == 1
            save(['Est_DTns_',ms{dj},'.mat'],'draws');           
        end    
    end
    disp(['data set ', num2str(repj), ' has completed!']);
    toc;
    disp(' ');
end












