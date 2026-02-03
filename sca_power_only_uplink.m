%% -------------------- Local function (SCA solver) --------------------
function [p_star, lat_hist, Ttot_u, sumP] = sca_power_only_uplink( ...
    B_u, h_u, n0, eta, d_bits_u, rho_u, Tenc_u, ...
    alpha_u, w_u, p_min_u, p_max_u, P_tot, ...
    sca, Qfun, dQdzfun, N)

    Nusers = numel(B_u);
    log2fun = @(x) log(1+x)/log(2);

    % Precompute constants
    a_u    = h_u ./ (B_u.*n0);                          
    C_tx_u = d_bits_u .* (1-rho_u) ./ (eta .* B_u);     

    % Exact latencies 
    Ttx_exact  = @(p) C_tx_u ./ log2fun(1 + a_u.*p);
    Ttot_exact = @(p) Tenc_u + Ttx_exact(p);

    % fmincon options
    opts = optimoptions('fmincon', ...
        'Algorithm','interior-point', 'Display','off', ...
        'StepTolerance',1e-12, 'OptimalityTolerance',1e-10, 'ConstraintTolerance',1e-10);

    % Init
    p_i = min(p_max_u, max(p_min_u, 0.1*P_tot/Nusers * ones(Nusers,1)));
    sc = sum(p_i)/P_tot; if sc > 1, p_i = p_i/sc; end

    lat_hist = zeros(1, sca.max_iter);

    % Helper function
    function g = get_gamma(it, Nloc, sca_struct)
        if isfield(sca_struct,'gamma')
            if isa(sca_struct.gamma,'function_handle')
                g = sca_struct.gamma(it, Nloc);
            else
                g = sca_struct.gamma; 
            end
        else
            g = 0.05; 
        end
        % clamp to (0,1]
        g = max(min(g,1.0), 1e-6);
    end

    for it = 1:sca.max_iter
        z_i = a_u .* p_i;
        Ai  = log(1 + z_i);
        Bi  = z_i ./ (1 + z_i);
        ln2 = log(2);

        % Linearized semantic constraints 
        Qi  = arrayfun(Qfun, z_i);
        dQi = arrayfun(dQdzfun, z_i);

        % Aineq * p <= bineq: [N semantic rows; 1 sum-power row]
        Aineq = zeros(Nusers+1, Nusers);
        bineq = zeros(Nusers+1, 1);

        for u = 1:Nusers
            if dQi(u) > 0
                Aineq(u, u) = -(dQi(u) * a_u(u));
                bineq(u)    = -(alpha_u(u) - Qi(u) + dQi(u)*z_i(u));
            else
                Aineq(u, u) = 0; bineq(u) = 0;
            end
        end
        Aineq(Nusers+1,:) = 1; bineq(Nusers+1) = P_tot;

        lb = p_min_u; ub = p_max_u;

        % Convex surrogate objective
        Cvec = C_tx_u; zi = z_i; avec = a_u; Bi_vec = Bi; Ai_vec = Ai;
        obj_hat = @(p) sum( w_u .* ( Tenc_u + (Cvec .* ln2) ./ (Ai_vec + Bi_vec .* (1 - zi ./ max(avec.*p, 1e-16))) ) );

        % Solve convex subproblem
        problem.objective = obj_hat;
        problem.x0        = p_i;
        problem.Aineq     = Aineq;
        problem.bineq     = bineq;
        problem.lb        = lb;
        problem.ub        = ub;
        problem.solver    = 'fmincon';
        problem.options   = opts;

        p_tilde = fmincon(problem);

        % === Use per-iteration
        gamma_t = get_gamma(it, N, sca);
        p_next  = min(ub, max(lb, (1-gamma_t)*p_i + gamma_t*p_tilde));

        % Log exact weighted sum latency
        lat_hist(it) = sum(w_u .* Ttot_exact(p_next));

        % Convergence check
        rel_impr = 1;
        if it > 1
            rel_impr = abs(lat_hist(it) - lat_hist(it-1)) / max(1e-12, lat_hist(it-1));
        end
        p_i = p_next;
        if (rel_impr <= sca.tol_rel) && (it >= sca.min_iters)
            lat_hist = lat_hist(1:it);
            break;
        end
    end

    % Outputs
    p_star = p_i;
    Ttot_u = Ttot_exact(p_star);
    sumP   = sum(p_star);
end
