% --- Black-Scholes PDE Solver: Explicit FDM ---
function [S, t, V] = explicit_black_scholes(S0, K, r, sigma, T, Smax, M, N, isCall)
    % S0: initial asset price, K: strike, r: rate, sigma: volatility, T: maturity
    % Smax: max asset price, M: asset steps, N: time steps, isCall: 1=call, 0=put
    dS = Smax / M;
    dt = T / N;
    S = linspace(0, Smax, M+1)';
    t = linspace(0, T, N+1);
    V = zeros(M+1, N+1);

    % Payoff at maturity
    if isCall
        V(:, end) = max(S - K, 0);
    else
        V(:, end) = max(K - S, 0);
    end

    % Boundary conditions
    for n = N:-1:1
        V(1, n) = 0;
        if isCall
            V(end, n) = Smax - K * exp(-r * (T - t(n)));
        else
            V(end, n) = 0;
        end
        for m = 2:M
            delta = (V(m+1, n+1) - V(m-1, n+1)) / (2*dS);
            gamma = (V(m+1, n+1) - 2*V(m, n+1) + V(m-1, n+1)) / (dS^2);
            V(m, n) = V(m, n+1) + dt * (0.5*sigma^2*S(m)^2*gamma + r*S(m)*delta - r*V(m, n+1));
        end
    end
end

% --- Black-Scholes PDE Solver: Implicit FDM ---
function [S, t, V] = implicit_black_scholes(S0, K, r, sigma, T, Smax, M, N, isCall)
    dS = Smax / M;
    dt = T / N;
    S = linspace(0, Smax, M+1)';
    t = linspace(0, T, N+1);
    V = zeros(M+1, N+1);

    % Payoff at maturity
    if isCall
        V(:, end) = max(S - K, 0);
    else
        V(:, end) = max(K - S, 0);
    end

    % Tridiagonal coefficients
    for n = N:-1:1
        A = zeros(M-1, M-1);
        B = zeros(M-1, 1);
        for m = 2:M
            j = m-1;
            a = 0.5*dt*(sigma^2*(m-1)^2 - r*(m-1));
            b = 1 + dt*(sigma^2*(m-1)^2 + r);
            c = -0.5*dt*(sigma^2*(m-1)^2 + r*(m-1));
            if j > 1
                A(j, j-1) = a;
            end
            A(j, j) = b;
            if j < M-1
                A(j, j+1) = c;
            end
            B(j) = V(m, n+1);
        end
        % Boundary conditions
        if isCall
            B(1) = B(1) - a * 0;
            B(end) = B(end) - c * (Smax - K * exp(-r * (T - t(n))));
        else
            B(1) = B(1) - a * 0;
            B(end) = B(end) - c * 0;
        end
        V(2:M, n) = A \ B;
        V(1, n) = 0;
        if isCall
            V(end, n) = Smax - K * exp(-r * (T - t(n)));
        else
            V(end, n) = 0;
        end
    end
end

% --- Black-Scholes PDE Solver: Crank-Nicolson FDM ---
function [S, t, V] = crank_nicolson_black_scholes(S0, K, r, sigma, T, Smax, M, N, isCall)
    dS = Smax / M;
    dt = T / N;
    S = linspace(0, Smax, M+1)';
    t = linspace(0, T, N+1);
    V = zeros(M+1, N+1);

    % Payoff at maturity
    if isCall
        V(:, end) = max(S - K, 0);
    else
        V(:, end) = max(K - S, 0);
    end

    % Tridiagonal system for Crank-Nicolson
    for n = N:-1:1
        A = zeros(M-1, M-1);
        B = zeros(M-1, 1);
        for m = 2:M
            j = m-1;
            alpha = 0.25*dt*(sigma^2*(m-1)^2 - r*(m-1));
            beta = -0.5*dt*(sigma^2*(m-1)^2 + r);
            gamma = 0.25*dt*(sigma^2*(m-1)^2 + r*(m-1));
            if j > 1
                A(j, j-1) = -alpha;
            end
            A(j, j) = 1 - beta;
            if j < M-1
                A(j, j+1) = -gamma;
            end
            % Right-hand side
            B(j) = alpha*V(m-1, n+1) + (1 + beta)*V(m, n+1) + gamma*V(m+1, n+1);
        end
        % Boundary conditions
        if isCall
            B(1) = B(1) + alpha * 0;
            B(end) = B(end) + gamma * (Smax - K * exp(-r * (T - t(n))));
        else
            B(1) = B(1) + alpha * 0;
            B(end) = B(end) + gamma * 0;
        end
        V(2:M, n) = A \ B;
        V(1, n) = 0;
        if isCall
            V(end, n) = Smax - K * exp(-r * (T - t(n)));
        else
            V(end, n) = 0;
        end
    end
end

% --- Bond Pricing with Term Structure PDE (Vasicek Model Example) ---
function [r_grid, t, V] = vasicek_bond_pricing(r0, a, b, sigma, T, r_max, M, N, F)
    % r0: initial rate, a: mean reversion, b: long-term mean, sigma: vol, T: maturity
    % r_max: max rate, M: rate steps, N: time steps, F: face value
    dr = r_max / M;
    dt = T / N;
    r_grid = linspace(0, r_max, M+1)';
    t = linspace(0, T, N+1);
    V = zeros(M+1, N+1);

    % Payoff at maturity
    V(:, end) = F;

    % Backward induction
    for n = N:-1:1
        for m = 2:M
            drift = a * (b - r_grid(m));
            diffusion = 0.5 * sigma^2;
            dVdr = (V(m+1, n+1) - V(m-1, n+1)) / (2*dr);
            d2Vdr2 = (V(m+1, n+1) - 2*V(m, n+1) + V(m-1, n+1)) / (dr^2);
            V(m, n) = V(m, n+1) + dt * (drift * dVdr + diffusion * d2Vdr2 - r_grid(m) * V(m, n+1));
        end
        V(1, n) = V(2, n);      % Neumann BC
        V(end, n) = V(end-1, n);
    end
end

% --- Mean-Variance Portfolio Optimization (Quadratic Programming) ---
function [w_opt, ret_opt, risk_opt] = mean_variance_optimization(mu, Sigma, target_ret)
    % mu: expected returns, Sigma: covariance matrix, target_ret: target portfolio return
    n = length(mu);
    H = 2*Sigma;
    f = zeros(n,1);
    Aeq = [mu'; ones(1,n)];
    beq = [target_ret; 1];
    lb = zeros(n,1);
    options = optimoptions('quadprog','Display','off');
    w_opt = quadprog(H, f, [], [], Aeq, beq, lb, [], [], options);
    ret_opt = mu' * w_opt;
    risk_opt = sqrt(w_opt' * Sigma * w_opt);
end

% --- Newton-Raphson for Derivative Calibration (e.g., implied volatility) ---
function sigma_star = calibrate_implied_vol(S, K, r, T, market_price, isCall)
    % S: spot, K: strike, r: rate, T: time, market_price: observed price
    % isCall: 1=call, 0=put
    tol = 1e-6;
    max_iter = 100;
    sigma = 0.2; % initial guess
    for i = 1:max_iter
        [price, vega] = bs_price_vega(S, K, r, sigma, T, isCall);
        diff = price - market_price;
        if abs(diff) < tol
            break;
        end
        sigma = sigma - diff / vega;
    end
    sigma_star = sigma;
end

function [price, vega] = bs_price_vega(S, K, r, sigma, T, isCall)
    d1 = (log(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T));
    d2 = d1 - sigma*sqrt(T);
    if isCall
        price = S * normcdf(d1) - K * exp(-r*T) * normcdf(d2);
    else
        price = K * exp(-r*T) * normcdf(-d2) - S * normcdf(-d1);
    end
    vega = S * normpdf(d1) * sqrt(T);
end

% --- Genetic Algorithm for Portfolio Optimization (Simple Example) ---
function [best_w, best_sharpe] = ga_portfolio_optimization(mu, Sigma, n_pop, n_gen, rf)
    % mu: expected returns, Sigma: covariance, n_pop: population size, n_gen: generations, rf: risk-free rate
    n = length(mu);
    pop = rand(n_pop, n);
    for i = 1:n_pop
        pop(i,:) = pop(i,:) / sum(pop(i,:));
    end
    best_sharpe = -Inf;
    best_w = [];
    for gen = 1:n_gen
        fitness = zeros(n_pop,1);
        for i = 1:n_pop
            w = pop(i,:)';
            ret = mu' * w;
            risk = sqrt(w' * Sigma * w);
            sharpe = (ret - rf) / risk;
            fitness(i) = sharpe;
            if sharpe > best_sharpe
                best_sharpe = sharpe;
                best_w = w;
            end
        end
        % Selection
        [~, idx] = sort(fitness, 'descend');
        pop = pop(idx(1:round(n_pop/2)), :);
        % Crossover & Mutation
        while size(pop,1) < n_pop
            parents = pop(randperm(size(pop,1),2), :);
            alpha = rand;
            child = alpha*parents(1,:) + (1-alpha)*parents(2,:);
            % Mutation
            if rand < 0.1
                m_idx = randi(n);
                child(m_idx) = abs(child(m_idx) + 0.1*randn);
            end
            child = child / sum(child);
            pop = [pop; child];
        end
    end
end

% --- Reinforcement Learning Placeholder for Portfolio Allocation ---
function [policy, rewards] = rl_portfolio_allocation(env, agent, episodes)
    % env: environment, agent: RL agent, episodes: number of episodes
    % Placeholder for RL-based portfolio allocation
    policy = [];
    rewards = [];
    % Implement RL logic here (e.g., Q-learning, DDPG, PPO)
end