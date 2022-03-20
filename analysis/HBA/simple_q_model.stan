data {
  int<lower=1> N;  // number of subjects
  int<lower=1> T;  // maximum number of trials
  int<lower=1> Tsubj[N];  // number of trials per subject
  int<lower=2> Nopt; // number of options
  int<lower=-1, upper=Nopt> choice[N, T];   // option selected
  int<lower=-1, upper=1> reward[N, T];  // outcome reward (0 or 1)
}

transformed data {
  vector[Nopt] initV;  // initial values for V
  initV = rep_vector(0.0, Nopt); 
}

parameters {
  // Group-level parameters
  vector[2] mu_pr;
  vector<lower=0>[2] sigma;

  // Subject-level parameters 
  vector[N] alpha_pr;
  vector[N] beta_pr;

}

transformed parameters {
  vector<lower=0, upper=1>[N] alpha;
  vector<lower=0, upper=10>[N] beta;
  
  alpha = Phi_approx(mu_pr[1] + sigma[1] * alpha_pr);
  beta = Phi_approx(mu_pr[2] + sigma[2] * beta_pr) * 10;
}

model {
  // Priors for group-level parameters
  mu_pr ~ normal(0, 1);
  sigma ~ normal(0, 0.2);
  
  // Priors for subject-level parameters
  alpha_pr ~ normal(0, 1);
  beta_pr ~ normal(0, 1);
  
  for (sbj in 1:N) {
    vector[Nopt] v; // action values
    real pe; // prediction error
    v = initV;
    
    for (t in 1:Tsubj[sbj]) {
      choice[sbj,t] ~ categorical_logit( beta[sbj] * v);
      pe = reward[sbj,t] - v[choice[sbj,t]];      
      v[choice[sbj, t]] += alpha[sbj] * pe; 
    }
  }
}

generated quantities {
  
  real log_lik[N];
  real y_pred[N, T];
  
  // posterior mean for group-level
  real<lower=0, upper=1>  mu_alpha;
  real<lower=0, upper=10> mu_beta;
  
  // Set all posterior predictions to 0 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      y_pred[i, t] = -1;
    }
  }
  
  mu_alpha = Phi_approx(mu_pr[1]);
  mu_beta = Phi_approx(mu_pr[2]) * 10;
  
  for (sbj in 1:N) {
    vector[Nopt] v; // action values
    real pe; // prediction error
    v = initV;
    log_lik[sbj] = 0.0;
    
    for (t in 1:Tsubj[sbj]) {
      log_lik[sbj] += categorical_logit_lpmf(choice[sbj,t] | beta[sbj] * v);
      y_pred[sbj, t] = categorical_rng(softmax(beta[sbj] * v));
      pe = reward[sbj,t] - v[choice[sbj,t]];      
      v[choice[sbj, t]] += alpha[sbj] * pe; 
    }
  }
}
