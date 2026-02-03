
clear; clc; rng(42);

%% -------------------- Global constants --------------------
eta       = 0.70;                   
n0_dBmHz  = -174;                   
n0        = 10^((n0_dBmHz-30)/10);  
Btot      = 2e6;                    
P_tot     = 2.0;                    
alpha_val = 0.95;                   
rho_k     = 0.3;                  
d_bits    = 50952*10;              
Tenc_val  = 1.0;                    

% Pathloss/fading model controls 
beta      = 3.2;                    
Dmin      = 20;                     
Dmax      = 67;                    
sigma_SH  = 6;                     
PL0_dB    = -40;                    

% SCA settings 
sca.max_iter   = 100;
sca.min_iters  = 8;
sca.tol_rel    = 1e-5;

N_list = [100 500 1000];

gam_map = containers.Map({N_list(1), N_list(2), N_list(3)}, {0.03, 0.025, 0.05}); 
sca.gamma = @(it, N) gam_map(N);

% Quality model 
q0 = 0.10; q1 = 0.40;
Qfun    = @(z) min(1, max(0, q0 + q1*log(1+z)/log(2)));
dQdzfun = @(z) q1 ./ ((1+z)*log(2));

% Scenarios to compare
user = N_list;

% Storage for convergence histories 
histories = cell(numel(N_list),1);
final_lat = zeros(numel(N_list),1);
final_avg = zeros(numel(N_list),1);
final_sumP= zeros(numel(N_list),1);

%% -------------------- Run SCA for each N --------------------
for idx = 1:numel(N_list)
    N = N_list(idx);
    B_u = (Btot/N) * ones(N,1);

    
    D_u       = Dmin + (Dmax-Dmin)*rand(N,1);               
    PL_dB_u   = PL0_dB - 10*beta*log10(D_u) + sigma_SH*randn(N,1);
    h_u       = 10.^(PL_dB_u/10);                           

    % ----- Per-user constants -----
    d_bits_u  = d_bits  * ones(N,1);
    rho_u     = rho_k   * ones(N,1);
    Tenc_u    = Tenc_val* ones(N,1);
    alpha_u   = alpha_val * ones(N,1);
    w_u       = ones(N,1);
    p_min_u   = 1e-6 * ones(N,1);
    p_max_u   = 0.8  * ones(N,1);

    % ----- Run SCA 
    [p_star, lat_hist, Ttot_u, sumP] = sca_power_only_uplink( ...
        B_u, h_u, n0, eta, d_bits_u, rho_u, Tenc_u, ...
        alpha_u, w_u, p_min_u, p_max_u, P_tot, ...
        sca, Qfun, dQdzfun, N);       

    % Store results
    histories{idx} = lat_hist;
    final_lat(idx) = sum(w_u .* Ttot_u);
    final_avg(idx) = mean(Ttot_u);
    final_sumP(idx)= sumP;

end

%% ===== SCA Convergence Across User Scales (Side-by-Side, Clean) =====
figure('Color','w','Position',[100 100 1200 420]);
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

names      = {'N = 100','N = 500','N = 1000'};
sca_colors = lines(3);
sca_marker = 'o';

for i = 1:3
    nexttile; hold on; grid on;
    ni = numel(histories{i});
    plot(1:ni, histories{i}, '-', ...
        'Color', sca_colors(i,:), 'LineWidth', 2.2, ...
        'DisplayName', sprintf('SCA (%s)', names{i}));
    plot(1:ni, histories{i}, 'LineStyle','none', 'Marker', sca_marker, ...
        'MarkerSize', 6, 'MarkerFaceColor', sca_colors(i,:), ...
        'MarkerEdgeColor','w', 'HandleVisibility','off', ...
        'MarkerIndices', 1:max(1,round(ni/20)):ni);
    yline(histories{i}(end), ':', 'Color', [0.45 0.75 0.35], ...
        'LineWidth', 1.4, 'HandleVisibility','off');
    xlabel('Iteration','FontSize',14);
    if i == 1
        ylabel('Latency T_{tot} (s)','FontSize',12);
    end
    title(names{i}, 'FontSize',14,'FontWeight','bold');
    set(gca,'FontSize',12,'LineWidth',1.2);
    pbaspect([1 1 1]);
    xlim([1 ni]);
end


