#!/usr/bin/env Rscript
# Joint model: which Grok prompt guideline(s) drive the bias?
#   pred ~ source * exclude_guideline + (1 | sentence_id) + (1 | gen_id)
# exclude_guideline = 0 (all four guidelines present) | g in {1..4} (guideline g removed).
# pred = bullet stance toward legal abortion (for = pro-choice, against = pro-life).
# source = source tweet's stance.

suppressPackageStartupMessages({
  library(brms)
  library(tidyverse)
  library(jsonlite)
})

input_dir  <- "outputs/judge_predict"
input_glob <- "judge_predict__topic=abortion__exclude_guideline=*.tsv"
output_dir <- "outputs/bayesian"
cache_dir  <- file.path(output_dir, "cache")
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

prepare_data <- function() {
  files <- Sys.glob(file.path(input_dir, input_glob))
  if (length(files) == 0) stop("No input files found.")
  bullet_cols <- c("prediction_bullet_1", "prediction_bullet_2", "prediction_bullet_3")

  dfs <- lapply(files, function(f) {
    eg <- as.integer(sub(".*exclude_guideline=([0-9]+)\\..*", "\\1", basename(f)))
    df <- read_tsv(f, show_col_types = FALSE, quote = "")
    df$exclude_guideline <- eg
    df
  })

  bind_rows(dfs) %>%
    select(sentence_id, output_id, exclude_guideline, annotation, all_of(bullet_cols)) %>%
    pivot_longer(all_of(bullet_cols), names_to = "bullet", values_to = "pred") %>%
    filter(pred %in% c("for", "against", "neutral"),
           annotation %in% c("Argument_for", "Argument_against")) %>%
    mutate(
      source = factor(if_else(annotation == "Argument_for", "prochoice", "prolife"),
                      levels = c("prochoice", "prolife")),
      pred   = factor(pred, levels = c("neutral", "for", "against")),
      exclude_guideline = factor(exclude_guideline, levels = c("0","1","2","3","4")),
      gen_id = paste(sentence_id, output_id, exclude_guideline, sep = "_")
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
    pred ~ source * exclude_guideline + (1 | sentence_id) + (1 | gen_id),
    data    = data,
    family  = categorical(refcat = "neutral"),
    prior   = priors,
    chains  = 4, iter = 4000, warmup = 2000, cores = 4,
    seed    = 42, silent = 2, refresh = 0,
    control = list(adapt_delta = 0.95)
  )
}

summarize_draws <- function(x) {
  q <- quantile(x, c(0.025, 0.975))
  list(mean = mean(x), lower_95 = unname(q[1]), upper_95 = unname(q[2]),
       prob_gt_0 = mean(x > 0))
}

extract_statistics <- function(fit, data) {
  egs <- levels(data$exclude_guideline)
  newdata <- expand.grid(
    source            = factor(c("prochoice", "prolife"), levels = levels(data$source)),
    exclude_guideline = factor(egs,                       levels = egs),
    KEEP.OUT.ATTRS = FALSE
  )
  epred <- posterior_epred(fit, newdata = newdata, re_formula = NA)  # draws x 10 x 3
  cats  <- dimnames(epred)[[3]]
  idx   <- function(src, eg) which(newdata$source == src & newdata$exclude_guideline == eg)

  predicted_probs <- lapply(setNames(egs, paste0("eg=", egs)), function(eg) list(
    prochoice = setNames(lapply(cats, function(k) summarize_draws(epred[, idx("prochoice", eg), k])), cats),
    prolife   = setNames(lapply(cats, function(k) summarize_draws(epred[, idx("prolife",   eg), k])), cats)
  ))

  beta_sup <- function(eg) epred[, idx("prolife",   eg), "against"] - epred[, idx("prochoice", eg), "for"]
  beta_opp <- function(eg) epred[, idx("prochoice", eg), "against"] - epred[, idx("prolife",   eg), "for"]

  bias_per_condition <- lapply(setNames(egs, paste0("eg=", egs)), function(eg) list(
    beta_sup = summarize_draws(beta_sup(eg)),
    beta_opp = summarize_draws(beta_opp(eg))
  ))

  # DiD: bias under all guidelines (eg=0) minus bias with g removed.
  # > 0 => removing g shrinks that bias => guideline g contributes to it.
  guideline_contributions <- lapply(setNames(c("1","2","3","4"), paste0("guideline_", 1:4)), function(g) list(
    delta_beta_sup = summarize_draws(beta_sup("0") - beta_sup(g)),
    delta_beta_opp = summarize_draws(beta_opp("0") - beta_opp(g))
  ))

  list(
    n_bullets               = nrow(data),
    n_sentence_ids          = n_distinct(data$sentence_id),
    n_generations           = n_distinct(data$gen_id),
    n_per_condition         = as.list(table(data$exclude_guideline)),
    n_per_pred              = as.list(table(data$pred)),
    predicted_probs         = predicted_probs,
    bias_per_condition      = bias_per_condition,
    guideline_contributions = guideline_contributions
  )
}

main <- function() {
  data <- prepare_data()
  cat(sprintf("Bullets: %d, sentence_ids: %d, generations: %d\n",
              nrow(data), n_distinct(data$sentence_id), n_distinct(data$gen_id)))

  cache_file <- file.path(cache_dir, "grok_bias__guideline_test.rds")
  json_file  <- file.path(output_dir, "grok_bias__guideline_test.json")
  files      <- Sys.glob(file.path(input_dir, input_glob))

  if (file.exists(cache_file) && file.mtime(cache_file) > max(file.mtime(files))) {
    cat("Loading cached fit...\n")
    fit <- readRDS(cache_file)
  } else {
    cat("Fitting model...\n")
    fit <- fit_model(data)
    saveRDS(fit, cache_file)
  }

  stats <- extract_statistics(fit, data)

  cat("\nBias per condition:\n")
  for (eg in c("0","1","2","3","4")) {
    b <- stats$bias_per_condition[[paste0("eg=", eg)]]
    cat(sprintf("  eg=%s  beta_sup = %+.3f [%+.3f, %+.3f]   beta_opp = %+.3f [%+.3f, %+.3f]\n",
                eg,
                b$beta_sup$mean, b$beta_sup$lower_95, b$beta_sup$upper_95,
                b$beta_opp$mean, b$beta_opp$lower_95, b$beta_opp$upper_95))
  }

  cat("\nGuideline contribution (eg=0 bias minus eg=g bias; >0 => g contributes):\n")
  for (g in c("1","2","3","4")) {
    d <- stats$guideline_contributions[[paste0("guideline_", g)]]
    cat(sprintf("  g=%s  delta_beta_sup = %+.3f [%+.3f, %+.3f] P(>0)=%.3f   delta_beta_opp = %+.3f [%+.3f, %+.3f] P(>0)=%.3f\n",
                g,
                d$delta_beta_sup$mean, d$delta_beta_sup$lower_95, d$delta_beta_sup$upper_95, d$delta_beta_sup$prob_gt_0,
                d$delta_beta_opp$mean, d$delta_beta_opp$lower_95, d$delta_beta_opp$upper_95, d$delta_beta_opp$prob_gt_0))
  }

  write_json(c(list(input_dir = input_dir), stats),
             json_file, pretty = TRUE, auto_unbox = TRUE)
  cat(sprintf("\nResults: %s\n", json_file))
}

main()
