# Test script for Chromatogram R6 class
# -------------------------------------
# Run from project root with:  Rscript scripts/test_chrom.R

source("R/Chromatogram.R")

set.seed(123)
n <- 200
time <- seq(0, 20, length.out = n)
true_params <- list(a = 100, b = 10, c = 2)

# Generate synthetic Gaussian-like chromatogram data with noise
intensity <- true_params$a *
  exp(-((time - true_params$b)^2) / (2 * true_params$c^2)) +
  rnorm(n, sd = 5)

chrom <- Chromatogram$new(time, intensity)

chrom$smooth(window = 15, poly = 3)
chrom$fit_gaussian()
chrom$baseline(method = "min")
chrom$auc(use = "smoothed")

# Save plot as PNG
chrom$plot(save_path = "outputs/figures/test_chrom.png")

# New functionality
chrom$find_peaks()
chrom$integrate_peaks()
chrom$summarize_peaks()

cat("✅ Chromatogram analysis complete. Plot saved in outputs/figures.\n")
