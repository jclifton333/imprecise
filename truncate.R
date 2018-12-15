library(ggplot2)
# Indeterminate credences (P_1, P_2), conditional lognormals given actions 0, 1
# P_1: mu(a_0)=1, sigma.sq(a_0)=1, mu(a_1)=1, sigma.sq(a_1)=2
# P_2: mu(a_0)=2, sigma.sq(a_0)=2, mu(a_1)=1/2, sigma.sq(a_1)=3/2

truncated.expectation = function(mu, sigma.sq, eta, mc.replicates=1e6){
  # Compute expectation of eta-truncated lognormal with params (mu, sigma.sq)
  
  x = rlnorm(mc.replicates, mu, sigma.sq)
  quantiles = quantile(x, probs=(1-eta))
  return(mean(x[x < quantiles]))
}


best.action.under.each.p = function(etas=seq(0.1, 0.5, 0.1)){
  for(eta in etas){
    u.0.0 = truncated.expectation(1, 1, eta)
    u.0.1 = truncated.expectation(1, 2, eta)
    u.1.0 = truncated.expectation(2, 2, eta)
    u.1.1 = truncated.expectation(0.5, 1.5, eta)
    print(paste(eta,  u.0.0, u.0.1, u.1.0, u.1.1, sep=" "))
  }
}


get.truncated.pdf = function(mu, sigma.sq, eta, mc.replicates=1e6){
  # Get truncated pdf
  x = rlnorm(mc.replicates, mu, sigma.sq)
  q = quantile(x, probs=(1-eta))
  truncated.pdf = function(y){
    if(y < q){
      return(dlnorm(y, mu, sigma.sq))
    }
    else{
      return(0)
    }
  }
  # truncated.pdf = Vectorize(truncated.pdf)
  return(Vectorize(truncated.pdf))
}


plot.truncated.pdfs = function(){
  # funcs = list(get.truncated.pdf(1, 1, 0.3), get.truncated.pdf(1, 2, 0.3), get.truncated.pdf(2, 2, 0.3),
  #              get.truncated.pdf(0.5, 1.5, 0.3))
  cols = c('red', 'yellow', 'black', 'blue')
  names = list('a0.1', 'a1.1', 'a1.2', 'a1.2')
  p = ggplot() + 
    stat_function(fun=get.truncated.pdf(1, 1, 0.3), aes(colour="a0.1")) + 
    stat_function(fun=get.truncated.pdf(1, 2, 0.3), aes(colour="a1.1")) + 
    stat_function(fun=get.truncated.pdf(2, 2, 0.3), aes(colour="a0.2")) + 
    stat_function(fun=get.truncated.pdf(0.5, 1.5, 0.3), aes(colour="a1.2")) + 
    scale_colour_manual("PDF", values=c("red", "yellow", "black", "blue")) +
    scale_x_continuous(name="Utility", limits=c(0, 10)) + 
    scale_y_continuous(name="Density", limits=c(0, 0.5))
  print(p)
}

plot.truncated.pdfs()


# set.seed(3)
# best.action.under.each.p()