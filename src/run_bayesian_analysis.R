#!/usr/bin/env Rscript
# Bayesian Linear Mixed Effects Analysis for Opinion Shift
# Fits per dataset/LLM/topic:
#   delta ~ 1 + direction + (1 | sentence_id) + (1 | prompt_id)

# Load required libraries
suppressPackageStartupMessages({
  library(brms)
  library(tidyverse)
  library(jsonlite)
  library(posterior)
  library(parallel)
})

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)
keep_flips <- "--keep-flips" %in% args

# Get project root (assume script is run from project root)
project_root <- getwd()

# Define paths
data_dir <- file.path(project_root, "outputs/predict_opinion")
output_dir <- file.path(project_root, "outputs/bayesian")
cache_dir <- file.path(output_dir, "cache")

# Create output directories
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

# Required columns for analysis
required_columns <- c("sentence_id", "prompt_id", "confidence_original",
                       "confidence_transformed", "prediction_original",
                       "prediction_transformed", "annotation")

# Parse predict_opinion filename into a named list of parameters.
parse_filename <- function(filename) {
  # Strip extension and prefix
  base <- sub("\\.tsv$", "", filename)
  parts <- strsplit(base, "__")[[1]]

  # First part should be the experiment name (predict_opinion)
  if (length(parts) < 2 || parts[1] != "predict_opinion") {
    return(NULL)
  }

  # Parse key=value pairs from remaining parts
  params <- list()
  for (part in parts[-1]) {
    kv <- strsplit(part, "=", fixed = TRUE)[[1]]
    if (length(kv) == 2) {
      params[[kv[1]]] <- kv[2]
    }
  }

  # Require at least dataset, task, model, topic, quantification_method
  if (!all(c("dataset", "task", "model", "topic", "quantification_method") %in% names(params))) {
    return(NULL)
  }

  params
}

# Validate that data has all required columns, stopping with an error if not.
validate_columns <- function(data, file_path) {
  missing <- setdiff(required_columns, colnames(data))
  if (length(missing) > 0) {
    stop(sprintf(
      "File '%s' is missing required columns: %s\nRequired: %s\nFound: %s",
      file_path,
      paste(missing, collapse = ", "),
      paste(required_columns, collapse = ", "),
      paste(colnames(data), collapse = ", ")
    ))
  }
  TRUE
}

# Fit a brms model, loading from cache if available and fresh.
fit_with_cache <- function(fit_func, data, cache_file, input_mtime) {
  if (file.exists(cache_file)) {
    cache_mtime <- file.mtime(cache_file)
    if (cache_mtime > input_mtime) {
      cat(sprintf("  Loading cached model from %s...\n", basename(cache_file)))
      return(readRDS(cache_file))
    } else {
      cat(sprintf("  Cache stale, re-fitting model...\n"))
    }
  }

  fit <- fit_func(data)
  saveRDS(fit, cache_file)
  fit
}

# Fit delta ~ 1 + direction + (1|sentence_id) + (1|prompt_id) with brms.
# Direction is +1 for Argument_for, -1 for Argument_against.
fit_model_direction <- function(data) {
  # Compute delta in [0, 1] scale (positive = shift toward "for")
  data$delta <- data$confidence_transformed - data$confidence_original

  # Direction from gold annotations
  data$direction <- ifelse(data$annotation == "Argument_for", 1, -1)

  # Ensure prompt_id is a factor
  data$prompt_id <- as.factor(data$prompt_id)

  # Set priors (weakly informative, scaled for [0, 1] confidence)
  priors <- c(
    prior(normal(0, 0.25), class = "Intercept"),
    prior(normal(0, 0.25), class = "b"),
    prior(exponential(4), class = "sd"),
    prior(exponential(4), class = "sigma")
  )

  # Fit the model
  fit <- brm(
    delta ~ 1 + direction + (1 | sentence_id) + (1 | prompt_id),
    data = data,
    family = gaussian(),
    prior = priors,
    chains = 4,
    iter = 4000,
    warmup = 2000,
    cores = 4,
    seed = 42,
    silent = 2,
    refresh = 0
  )

  fit
}

# Extract posterior statistics from direction model for JSON export.
extract_statistics_direction <- function(fit, data) {
  # Extract posterior samples
  posterior_samples <- as_draws_df(fit)

  # Get R-squared
  r2 <- bayes_R2(fit)

  # Intercept statistics
  intercept <- list(
    mean = mean(posterior_samples$b_Intercept),
    lower_95 = unname(quantile(posterior_samples$b_Intercept, 0.025)),
    upper_95 = unname(quantile(posterior_samples$b_Intercept, 0.975)),
    prob_positive = mean(posterior_samples$b_Intercept > 0)
  )

  # Direction coefficient
  direction_coef <- list(
    mean = mean(posterior_samples$b_direction),
    lower_95 = unname(quantile(posterior_samples$b_direction, 0.025)),
    upper_95 = unname(quantile(posterior_samples$b_direction, 0.975)),
    prob_positive = mean(posterior_samples$b_direction > 0)
  )

  # Group effects: originally_for = intercept + direction, originally_against = intercept - direction
  originally_for <- posterior_samples$b_Intercept + posterior_samples$b_direction
  originally_against <- posterior_samples$b_Intercept - posterior_samples$b_direction

  group_effects <- list(
    originally_for = list(
      mean = mean(originally_for),
      lower_95 = unname(quantile(originally_for, 0.025)),
      upper_95 = unname(quantile(originally_for, 0.975)),
      prob_positive = mean(originally_for > 0)
    ),
    originally_against = list(
      mean = mean(originally_against),
      lower_95 = unname(quantile(originally_against, 0.025)),
      upper_95 = unname(quantile(originally_against, 0.975)),
      prob_positive = mean(originally_against > 0)
    )
  )

  # Random effects
  random_effects <- list(
    sd_sentence_id__Intercept = list(
      mean = mean(posterior_samples$sd_sentence_id__Intercept),
      lower_95 = unname(quantile(posterior_samples$sd_sentence_id__Intercept, 0.025)),
      upper_95 = unname(quantile(posterior_samples$sd_sentence_id__Intercept, 0.975))
    ),
    sd_prompt_id__Intercept = list(
      mean = mean(posterior_samples$sd_prompt_id__Intercept),
      lower_95 = unname(quantile(posterior_samples$sd_prompt_id__Intercept, 0.025)),
      upper_95 = unname(quantile(posterior_samples$sd_prompt_id__Intercept, 0.975))
    )
  )

  # Sigma (residual SD)
  sigma <- list(
    mean = mean(posterior_samples$sigma),
    lower_95 = unname(quantile(posterior_samples$sigma, 0.025)),
    upper_95 = unname(quantile(posterior_samples$sigma, 0.975))
  )

  # R-squared
  r2_stats <- list(
    mean = r2[1, "Estimate"],
    lower_95 = r2[1, "Q2.5"],
    upper_95 = r2[1, "Q97.5"]
  )

  list(
    n_observations = nrow(data),
    n_originally_for = sum(data$direction == 1),
    n_originally_against = sum(data$direction == -1),
    n_groups = n_distinct(data$sentence_id),
    intercept = intercept,
    direction = direction_coef,
    group_effects = group_effects,
    random_effects = random_effects,
    sigma = sigma,
    r2 = r2_stats
  )
}

# Process a single predict_opinion TSV file: fit model and save results as JSON.
process_file <- function(file_path, keep_flips = FALSE) {
  filename <- basename(file_path)

  # Parse filename
  parsed <- parse_filename(filename)
  if (is.null(parsed)) {
    return(list(
      status = "skipped",
      reason = "Filename does not match expected pattern"
    ))
  }

  dataset <- parsed$dataset
  task <- parsed$task
  model <- parsed$model
  topic <- parsed$topic
  quantification_method <- parsed$quantification_method

  cat(sprintf("\n%s\n", paste(rep("=", 80), collapse = "")))
  cat(sprintf("Processing: dataset=%s, task=%s, model=%s, topic=%s, method=%s\n", dataset, task, model, topic, quantification_method))
  cat(sprintf("Input: %s\n", file_path))

  # Define output paths (include task, quantification_method and flip flag to avoid collisions)
  flip_suffix <- ifelse(keep_flips, "__keep_flips", "")
  output_name <- sprintf("%s__%s__%s__%s__%s%s", dataset, task, model, topic, quantification_method, flip_suffix)
  cache_file_direction <- file.path(cache_dir, sprintf("%s__direction.rds", output_name))
  json_file <- file.path(output_dir, sprintf("%s.json", output_name))

  # Skip if output already exists
  if (file.exists(json_file)) {
    cat(sprintf("Output already exists: %s — skipping\n", json_file))
    return(list(
      status = "skipped",
      reason = "Output file already exists"
    ))
  }

  # Read data
  cat("Reading data...\n")
  data <- read_tsv(file_path, show_col_types = FALSE, quote = "")

  # Validate columns
  tryCatch({
    validate_columns(data, file_path)
  }, error = function(e) {
    stop(e$message)
  })

  n_total <- nrow(data)
  n_groups_total <- n_distinct(data$sentence_id)
  cat(sprintf("Data: %d observations, %d groups\n", n_total, n_groups_total))

  # === Opinion-flip statistics and optional filtering ===
  flipped <- data$prediction_original != data$prediction_transformed
  n_flipped <- sum(flipped)

  # Compute flip stats by annotation group
  is_for <- data$annotation == "Argument_for"
  is_against <- data$annotation == "Argument_against"
  n_for_total <- sum(is_for)
  n_against_total <- sum(is_against)
  n_for_flipped <- sum(flipped & is_for)
  n_against_flipped <- sum(flipped & is_against)

  cat(sprintf("Opinion flips: %d/%d (%.1f%%)\n",
              n_flipped, n_total, 100 * n_flipped / n_total))
  cat(sprintf("  For-texts flipped:     %d/%d (%.1f%%)\n",
              n_for_flipped, n_for_total,
              ifelse(n_for_total > 0, 100 * n_for_flipped / n_for_total, 0)))
  cat(sprintf("  Against-texts flipped: %d/%d (%.1f%%)\n",
              n_against_flipped, n_against_total,
              ifelse(n_against_total > 0, 100 * n_against_flipped / n_against_total, 0)))

  # Store flip statistics for JSON output
  flip_stats <- list(
    n_total = n_total,
    n_flipped = n_flipped,
    n_kept = n_total - n_flipped,
    pct_flipped = round(100 * n_flipped / n_total, 2),
    n_for_total = n_for_total,
    n_for_flipped = n_for_flipped,
    pct_for_flipped = round(ifelse(n_for_total > 0, 100 * n_for_flipped / n_for_total, 0), 2),
    n_against_total = n_against_total,
    n_against_flipped = n_against_flipped,
    pct_against_flipped = round(ifelse(n_against_total > 0, 100 * n_against_flipped / n_against_total, 0), 2),
    filter_applied = !keep_flips
  )

  if (keep_flips) {
    cat("Keeping all rows (--keep-flips enabled)\n")
  } else {
    data <- data[!flipped, ]
    cat(sprintf("After filtering: %d observations, %d groups\n", nrow(data), n_distinct(data$sentence_id)))
  }

  # Get input file modification time for cache validation
  input_mtime <- file.mtime(file_path)

  # === Fit Model: delta ~ 1 + direction + (1 | sentence_id) + (1 | prompt_id) ===
  cat("Fitting model: delta ~ 1 + direction + (1 | sentence_id) + (1 | prompt_id)...\n")
  fit_direction <- fit_with_cache(fit_model_direction, data, cache_file_direction, input_mtime)

  # Prepare data for statistics extraction (compute delta, direction from gold annotations)
  data_direction <- data %>%
    mutate(
      delta = confidence_transformed - confidence_original,
      direction = ifelse(annotation == "Argument_for", 1, -1)
    )

  cat("Extracting statistics (direction model)...\n")
  stats_direction <- extract_statistics_direction(fit_direction, data_direction)

  # Log direction model results
  cat(sprintf("  Intercept: %.4f [%.4f, %.4f], P(>0)=%.3f\n",
              stats_direction$intercept$mean,
              stats_direction$intercept$lower_95,
              stats_direction$intercept$upper_95,
              stats_direction$intercept$prob_positive))
  cat(sprintf("  Direction: %.4f [%.4f, %.4f], P(>0)=%.3f\n",
              stats_direction$direction$mean,
              stats_direction$direction$lower_95,
              stats_direction$direction$upper_95,
              stats_direction$direction$prob_positive))
  cat(sprintf("  Originally for:     %.4f [%.4f, %.4f]\n",
              stats_direction$group_effects$originally_for$mean,
              stats_direction$group_effects$originally_for$lower_95,
              stats_direction$group_effects$originally_for$upper_95))
  cat(sprintf("  Originally against: %.4f [%.4f, %.4f]\n",
              stats_direction$group_effects$originally_against$mean,
              stats_direction$group_effects$originally_against$lower_95,
              stats_direction$group_effects$originally_against$upper_95))

  # === Build JSON output ===
  output <- list(
    dataset = dataset,
    task = task,
    model = model,
    topic = topic,
    quantification_method = quantification_method,
    keep_flips = keep_flips,
    input_file = file_path,
    opinion_flip_filter = flip_stats,
    model_direction = list(
      description = "Direction model measuring differential effects based on original opinion polarity",
      formula = "delta ~ 1 + direction + (1 | sentence_id) + (1 | prompt_id)",
      direction_coding = "+1 for Argument_for, -1 for Argument_against (gold annotation)",
      n_observations = stats_direction$n_observations,
      n_originally_for = stats_direction$n_originally_for,
      n_originally_against = stats_direction$n_originally_against,
      n_groups = stats_direction$n_groups,
      intercept = stats_direction$intercept,
      direction = stats_direction$direction,
      group_effects = stats_direction$group_effects,
      random_effects = stats_direction$random_effects,
      sigma = stats_direction$sigma,
      r2 = stats_direction$r2
    )
  )

  # Write JSON
  cat(sprintf("Writing results to: %s\n", json_file))
  write_json(output, json_file, pretty = TRUE, auto_unbox = TRUE)

  list(
    status = "success",
    dataset = dataset,
    task = task,
    model = model,
    topic = topic,
    quantification_method = quantification_method,
    json_file = json_file
  )
}

# Main execution
main <- function() {
  cat(paste(rep("=", 80), collapse = ""), "\n")
  cat("Bayesian Analysis for Opinion Shift - Batch Processing\n")
  cat("Model: delta ~ 1 + direction + (1 | sentence_id) + (1 | prompt_id)\n")
  cat(paste(rep("=", 80), collapse = ""), "\n")
  cat(sprintf("Keep flips: %s\n", ifelse(keep_flips, "yes", "no")))
  cat(sprintf("Data directory: %s\n", data_dir))
  cat(sprintf("Output directory: %s\n", output_dir))
  cat(sprintf("Cache directory: %s\n", cache_dir))

  # Scan for TSV files
  files <- list.files(data_dir, pattern = "\\.tsv$", full.names = TRUE)

  if (length(files) == 0) {
    cat("\nNo TSV files found in data directory.\n")
    return(invisible(NULL))
  }

  cat(sprintf("\nFound %d TSV files\n", length(files)))

  # Process files in parallel (4 files × 4 cores per model = up to 16 cores)
  n_parallel_files <- 4
  cat(sprintf("Processing %d files in parallel (%d at a time)\n", length(files), n_parallel_files))

  results <- mclapply(files, function(file_path) {
    tryCatch({
      process_file(file_path, keep_flips = keep_flips)
    }, error = function(e) {
      list(status = "error", message = e$message, file = basename(file_path))
    })
  }, mc.cores = n_parallel_files)

  names(results) <- files

  # Summary
  cat(sprintf("\n%s\n", paste(rep("=", 80), collapse = "")))
  cat("SUMMARY\n")
  cat(paste(rep("=", 80), collapse = ""), "\n")

  success_count <- sum(sapply(results, function(x) x$status == "success"))
  error_count <- sum(sapply(results, function(x) x$status == "error"))
  skipped_count <- sum(sapply(results, function(x) x$status == "skipped"))

  cat(sprintf("Processed: %d files\n", length(files)))
  cat(sprintf("  Success: %d\n", success_count))
  cat(sprintf("  Errors: %d\n", error_count))
  cat(sprintf("  Skipped: %d\n", skipped_count))

  if (success_count > 0) {
    cat(sprintf("\nJSON results written to: %s\n", output_dir))
  }

  if (error_count > 0) {
    cat("\nFiles with errors:\n")
    for (name in names(results)) {
      if (results[[name]]$status == "error") {
        cat(sprintf("  - %s: %s\n", basename(name), results[[name]]$message))
      }
    }
  }

  invisible(results)
}

# Run main function
main()
