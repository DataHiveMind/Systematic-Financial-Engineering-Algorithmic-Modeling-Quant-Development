% --- Euler-Maruyama Simulation for Vasicek Model ---
function [r, t] = simulate_vasicek(r0, a, b, sigma, T, N, n_paths)
    % r0: initial rate, a: mean reversion, b: long-term mean, sigma: vol
    % T: time horizon, N: time steps, n_paths: number of simulations
    dt = T / N;
    t = linspace(0, T, N+1);
    r = zeros(n_paths, N+1);
    r(:,1) = r0;
    for i = 1:n_paths
        for n = 1:N
            dW = sqrt(dt) * randn;
            r(i, n+1) = r(i, n) + a * (b - r(i, n)) * dt + sigma * dW;
        end
    end
end

% --- Euler-Maruyama Simulation for CIR Model ---
function [r, t] = simulate_cir(r0, a, b, sigma, T, N, n_paths)
    dt = T / N;
    t = linspace(0, T, N+1);
    r = zeros(n_paths, N+1);
    r(:,1) = r0;
    for i = 1:n_paths
        for n = 1:N
            dW = sqrt(dt) * randn;
            r(i, n+1) = abs(r(i, n) + a * (b - r(i, n)) * dt + sigma * sqrt(max(r(i, n),0)) * dW);
        end
    end
end

% --- Monte Carlo Forecast for Interest Rate Paths ---
function [paths, t] = monte_carlo_rates(model, params, r0, T, N, n_paths)
    % model: 'vasicek' or 'cir'
    if strcmpi(model, 'vasicek')
        [paths, t] = simulate_vasicek(r0, params.a, params.b, params.sigma, T, N, n_paths);
    elseif strcmpi(model, 'cir')
        [paths, t] = simulate_cir(r0, params.a, params.b, params.sigma, T, N, n_paths);
    else
        error('Unknown model type');
    end
end

% --- Maximum Likelihood Estimation (MLE) for Vasicek Model ---
function [a_hat, b_hat, sigma_hat] = mle_vasicek(r, dt)
    % r: observed rates (vector), dt: time step
    n = length(r) - 1;
    x = r(1:end-1);
    y = r(2:end);
    Sx = sum(x);
    Sy = sum(y);
    Sxx = sum(x.^2);
    Sxy = sum(x .* y);
    Syy = sum(y.^2);

    theta = (n*Sxy - Sx*Sy) / (n*Sxx - Sx^2);
    mu = (Sy - theta*Sx) / (n - theta*n);
    a_hat = -log(theta) / dt;
    b_hat = mu;
    sigma_hat = sqrt((1/n) * sum((y - theta*x - (1-theta)*mu).^2) * 2*a_hat/(1-theta^2));
end

% --- Maximum Likelihood Estimation (MLE) for CIR Model (approximate) ---
function [a_hat, b_hat, sigma_hat] = mle_cir(r, dt)
    % Approximate MLE for CIR model
    n = length(r) - 1;
    x = r(1:end-1);
    y = r(2:end);
    dx = y - x;
    Sx = sum(x);
    Sdx = sum(dx);
    Sxx = sum(x.^2);
    Sxdx = sum(x .* dx);

    a_hat = -Sxdx / (dt * (Sxx - Sx^2/n));
    b_hat = Sdx/(a_hat*dt*n) + Sx/n;
    sigma_hat = sqrt(sum((dx - a_hat*(b_hat - x)*dt).^2) / (n*dt));
end