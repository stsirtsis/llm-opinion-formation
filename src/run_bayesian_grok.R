#!/usr/bin/env Rscript
# Bayesian test for asymmetric Grok bias on contextualization bullets (baseline,
# exclude_guideline=0):  pred ~ source + (1 | sentence_id) + (1 | gen_id)
# pred = bullet stance toward legal abortion (for = pro-choice, against = pro-life).
# source = source tweet's stance. gen_id ties the 3 bullets from one Grok generation.

suppressPackageStartupMessages({
  library(brms)
  library(tidyverse)
  library(jsonlite)
})

input_file <- "outputs/judge_predict/judge_predict__topic=abortion__exclude_guideline=0.tsv"
output_dir <- "outputs/bayesian"
cache_dir  <- file.path(output_dir, "cache")
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

prepare_data <- function(file_path) {
  bullet_cols <- c("prediction_bullet_1", "prediction_bullet_2", "prediction_bullet_3")
  read_tsv(file_path, show_col_types = FALSE, quote = "") %>%
    select(sentence_id, output_id, annotation, all_of(bullet_cols)) %>%
    pivot_longer(all_of(bullet_cols), names_to = "bullet", values_to = "pred") %>%
    filter(pred %in% c("for", "against", "neutral"),
           annotation %in% c("Argument_for", "Argument_against")) %>%
    mutate(
      source = factor(if_else(annotation == "Argument_for", "prochoice", "prolife"),
                      levels = c("prochoice", "prolife")),
      pred   = factor(pred, levels = c("neutral", "for", "against")),
      gen_id = paste(sentence_id, output_id, sep = "_")
    )
}

fit_model <- function(data) {
  priors <- c(
    prior(normal(0, 1.5), class = "Intercept", dpar = "mufor"),
    prior(normal(0, 1.5), class = "Intercept", dpar = "muagainst"),
    prior(normal(0, 1.5), class = "b",         dpar = "mufor"),
    prior(normal(0, 1.5), class = "b",         dpar = "muagainst"),
    prior(exponential(2), class = "sd",        dpar = "mufor"),
    prior(exponential(2), class = "sd",        dpar = "muagainst")
  )
  brm(
    pred ~ source + (1 | sentence_id) + (1 | gen_id),
    data    = data,
    family  = categorical(refcat = "neutral"),
    prior   = priors,
    chains  = 4, iter = 4000, warmup = 2000, cores = 4,
    seed    = 42, silent = 2, refresh = 0
  )
}

summarize_draws <- function(x) {
  q <- quantile(x, c(0.025, 0.975))
  list(mean = mean(x), lower_95 = unname(q[1]), upper_95 = unname(q[2]),
       prob_gt_0 = mean(x > 0))
}

extract_statistics <- function(fit, data) {
  newdata <- data.frame(source = factor(c("prochoice", "prolife"),
                                        levels = levels(data$source)))
  epred <- posterior_epred(fit, newdata = newdata, re_formula = NA)  # draws x source x category
  cats  <- dimnames(epred)[[3]]

  predicted_probs <- list(
    prochoice = setNames(lapply(cats, function(k) summarize_draws(epred[, 1, k])), cats),
    prolife   = setNames(lapply(cats, function(k) summarize_draws(epred[, 2, k])), cats)
  )

  # beta_sup = P(claim prolife | post prolife)   - P(claim prochoice | post prochoice)
  #          > 0 => Grok supports pro-life posts more than pro-choice posts.
  # beta_opp = P(claim prolife | post prochoice) - P(claim prochoice | post prolife)
  #          > 0 => Grok opposes pro-choice posts more than pro-life posts.
  bias_contrasts <- list(
    beta_sup = summarize_draws(epred[, 2, "against"] - epred[, 1, "for"]),
    beta_opp = summarize_draws(epred[, 1, "against"] - epred[, 2, "for"])
  )

  list(
    n_bullets       = nrow(data),
    n_sentence_ids  = n_distinct(data$sentence_id),
    n_generations   = n_distinct(data$gen_id),
    n_per_source    = as.list(table(data$source)),
    n_per_pred      = as.list(table(data$pred)),
    predicted_probs = predicted_probs,
    bias_contrasts  = bias_contrasts
  )
}

main <- function() {
  data <- prepare_data(input_file)
  cat(sprintf("Bullets: %d, sentence_ids: %d, generations: %d\n",
              nrow(data), n_distinct(data$sentence_id), n_distinct(data$gen_id)))

  cache_file <- file.path(cache_dir, "grok_bias__source_test.rds")
  json_file  <- file.path(output_dir, "grok_bias__source_test.json")

  if (file.exists(cache_file) && file.mtime(cache_file) > file.mtime(input_file)) {
    cat("Loading cached fit...\n")
    fit <- readRDS(cache_file)
  } else {
    cat("Fitting model...\n")
    fit <- fit_model(data)
    saveRDS(fit, cache_file)
  }

  stats <- extract_statistics(fit, data)
  bc <- stats$bias_contrasts
  cat(sprintf("beta_sup = %.3f [%.3f, %.3f], P(>0) = %.3f\n",
              bc$beta_sup$mean, bc$beta_sup$lower_95, bc$beta_sup$upper_95, bc$beta_sup$prob_gt_0))
  cat(sprintf("beta_opp = %.3f [%.3f, %.3f], P(>0) = %.3f\n",
              bc$beta_opp$mean, bc$beta_opp$lower_95, bc$beta_opp$upper_95, bc$beta_opp$prob_gt_0))

  write_json(c(list(input_file = input_file), stats),
             json_file, pretty = TRUE, auto_unbox = TRUE)
  cat(sprintf("Results: %s\n", json_file))
}

main()
