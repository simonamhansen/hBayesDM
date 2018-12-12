#include /pre/license.stan

data {
  // declares N, T, Tsubj[N]
#include /data/NT.stan
  // declares delay_later[N, T], amount_later[N, T], delay_sooner[N, T], amount_sooner[N, T]
#include /data/dd.stan
  // declares choice[N, T];
#include /data/chunks/choice_NT_01.stan
}

parameters {
// Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters
  vector[2] mu_pr;
  vector<lower=0>[2] sigma;

  // Subject-level raw parameters (for Matt trick)
  vector[N] r_pr;
  vector[N] beta_pr;
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0, upper=1>[N] r;
  vector<lower=0, upper=5>[N] beta;

  for (i in 1:N) {
    r[i]    = Phi_approx(mu_pr[1] + sigma[1] * r_pr[i]);
    beta[i] = Phi_approx(mu_pr[2] + sigma[2] * beta_pr[i]) * 5;
  }
}

model {
// Exponential function
  // Hyperparameters
  mu_pr  ~ normal(0, 1);
  sigma ~ normal(0, 0.2);

  // individual parameters
  r_pr    ~ normal(0, 1);
  beta_pr ~ normal(0, 1);

  for (i in 1:N) {
    // Define values
    real ev_later;
    real ev_sooner;

    for (t in 1:(Tsubj[i])) {
      ev_later  = amount_later[i, t]  * exp(-1 * r[i] * delay_later[i, t]);
      ev_sooner = amount_sooner[i, t] * exp(-1 * r[i] * delay_sooner[i, t]);
      choice[i, t] ~ bernoulli_logit(beta[i] * (ev_later - ev_sooner));
    }
  }
}
generated quantities {
  // For group level parameters
  real<lower=0, upper=1> mu_r;
  real<lower=0, upper=5> mu_beta;

  // For log likelihood calculation
  real log_lik[N];

  // For posterior predictive check
  real y_pred[N, T];

  // Set all posterior predictions to 0 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      y_pred[i, t] = -1;
    }
  }

  mu_r    = Phi_approx(mu_pr[1]);
  mu_beta = Phi_approx(mu_pr[2]) * 5;

  { // local section, this saves time and space
    for (i in 1:N) {
      // Define values
      real ev_later;
      real ev_sooner;

      log_lik[i] = 0;

      for (t in 1:(Tsubj[i])) {
        ev_later  = amount_later[i, t]  * exp(-1 * r[i] * delay_later[i, t]);
        ev_sooner = amount_sooner[i, t] * exp(-1 * r[i] * delay_sooner[i, t]);
        log_lik[i] += bernoulli_logit_lpmf(choice[i, t] | beta[i] * (ev_later - ev_sooner));

        // generate posterior prediction for current trial
        y_pred[i, t] = bernoulli_rng(inv_logit(beta[i] * (ev_later - ev_sooner)));
      }
    }
  }
}
