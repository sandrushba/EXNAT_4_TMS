#' Get Dprime (d')
#'
#' Computes Signal Detection Theory index d'
#'
#' @param n_hit Number of hits.
#' @param n_fa Number of false alarms.
#' @param n_miss Number of misses.
#' @param n_cr Number of correct rejections.
#' 
#' @return Calculates the d-prime for a given vector with hits, misses, false alarms & correct rejections.
#'
#' Returns the following index:
#' \itemize{
#'  \item{\strong{dprime (d')}: }{The response sensitivity index. Is calculated by sutracting the Z-value of the false-alarm rate from the Z-value of the hit-rate.}
#'  }
#'
#' Note that for d', adjustement for extreme values (hit or false alarm rates of 0 or 1) are made by replacing them by 0.01 or 0.99, respectively, following one the recommandations of Macmillan & Creelman (1991).
#'
#' @importFrom stats qnorm
#' @export


get_dprime <- function(nback_responses) {
  
  # I'll use the d'-prime (= detection prime) value as described 
  # in the Haatveit et al. (2010) paper (p. 873, right text column):
  # https://doi.org/10.1080/13803391003596421
  
  # d' = Z(hit rate) – Z(false alarm rate)
  
  # hit rate = how many hits were there when a target was presented?
  #            --> hit rate = number of hits / (number of hits + number of misses)
  
  # false alarm rate = how many false alarms were there when no target was presented?
  #            --> false alarm rate = number of false alarms / (number of false alarms + number of correct rejections)
  
  ############################## 
  
  # exctract nr of hits, misses, false alarms and correct rejections
  n_hits <- length(which(nback_responses == "hit"))
  n_fa   <- length(which(nback_responses == "false alarm"))
  n_miss <- length(which(nback_responses == "miss"))
  n_cr   <- length(which(nback_responses == "correct rejection"))
  
  # get hit rate & false alarm rate
  hit_rate <- n_hits / (n_miss + n_hits)
  fa_rate <- n_fa / (n_cr + n_fa)
  
  ############################## 
  
  # Adjust hit rate & false alarm rate so we don't get problems with extreme values
  # (i.e. when hit or false alarm rate = 0 or 100)
  
  # There are multiple ways you can adjust the hit & false alarm rates:
  
  # Option 1: Use a non-parametric statistic such as A' instead of d' (Craig, 1979) 
  # --> Nope, I want the more widely known d-prime.
  
  # Option 2: Aggregate data from multiple subjects before calculating 
  #           the statistic (Macmillan & Kaplan, 1985)
  # --> Nope, I want individual d-primes.
  
  # Option 3: Add 0.5 to both the number of hits and the number of false alarms, 
  # and add 1 to both the number of signal trials and the number of noise trials; 
  # dubbed the loglinear approach (Hautus, 1995)
  # --> Nope, only suitable for equal target to non-target-ratios.
  #     Here's a nice explanation in Jeff's answer:
  #     https://stats.stackexchange.com/questions/134779/d-prime-with-100-hit-rate-probability-and-0-false-alarm-probability
  
  # Option 4: Adjust only the extreme values by replacing rates of 0 with 0.5 / n
  # and rates of 1 with (n - 0.5) / n 
  # where n is the number of signal or noise trials (Macmillan & Kaplan, 1985)
  # --> Problem: This is not working well with a low number of trials.
  
  # Option 5: Define 0/0 to be 0 (Fienberg, 1977: 109)
  # --> What. No.
  
  # Option 6: Set hit or false alarm rates of 0 [1] to 0.01 [0.99].
  # --> I like this one best.
  # Sources for this approach:
  # From Iverson (1996; doi:10.1121/1.415234): "The z transform
  # reaches infinity when percentages equal 0 or 100, so tokens
  # with 0% /l/ identifications were assigned values of 1% and
  # tokens with 100% /l/ identifications were assigned values of
  # 99% (Macmillan and Creelman, 1991)."
  # They cite Macmillan and Creelman, 1991, 
  # Chapter 1 (The Yes-No Experiment: Sensitivity), p. 8, penultimate paragraph.
  # PDF: https://fisica.cab.cnea.gov.ar/escuelaib2014-neurociencias/restricted/CURSOS/GUILLERMO/Detection%20Theory.pdf

  ##############################  

  # Adjustment as described for Option 4:
  # # adjust extreme hit rates
  # if (hit_rate == 0){ 
  #   n <- (n_miss + n_hits) # n = number of target trials
  #   hit_rate <- 0.5 / n
  # } else if (hit_rate == 1){ 
  #   n <- (n_miss + n_hits) # n = number of target trials
  #   hit_rate <- (n - 0.5) / n 
  # }
  # 
  # # adjust extreme false-alarm rates
  # if (fa_rate == 0){ 
  #   n <- n_cr + n_fa # n = number of non-target trials
  #   fa_rate <- 0.5 / n
  # } else if (fa_rate == 1){ 
  #   n <- n_cr + n_fa # n = number of non-target trials
  #   fa_rate <- (n - 0.5) / n 
  # }
  
  # Adjustment as described for Option 6:
  # adjust extreme hit rates
  if (hit_rate == 0){ 
    hit_rate <- 0.01 # take 1% hit rate instead of 0%
  } else if (hit_rate == 1){ 
    hit_rate <- 0.99 # take 99% hit rate instead of 0%
  }
  # adjust extreme false-alarm rates
  if (fa_rate == 0){ 
    fa_rate <- 0.01 # take 1% f.a. rate instead of 0%
  } else if (fa_rate == 1){ 
    fa_rate <- 0.99 # take 99% f.a. rate instead of 0%
  }
  
  ############################## 
  
  # get z-values of hit rate & false alarm rate and subtract false alarms from hits.
  dprime <- qnorm(hit_rate) - qnorm(fa_rate)
  
  # return d-prime:
  return(dprime)
  
  # Interpretation: 
  # highest possible d-prime (only correct responses): d =  4.652696
  # lowest possible d-prime (only wrong responses):    d = -4.652696
  # participant always or never pressed the button:    d =  0
  
}# END d-prime function
