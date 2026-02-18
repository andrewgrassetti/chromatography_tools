# Load required packages
library(R6)
library(ggplot2)
library(signal)
library(minpack.lm)
library(pracma)

`%||%` <- function(a, b) if (!is.null(a)) a else b

Chromatogram <- R6Class("Chromatogram",
  public = list(
    time = NULL,
    intensity = NULL,
    smoothed = NULL,
    fit_params = NULL,
    baseline_subtracted = NULL,
    peaks = NULL,

    initialize = function(time, intensity) {
      stopifnot(length(time) == length(intensity))
      self$time <- as.numeric(time)
      self$intensity <- as.numeric(intensity)
      invisible(self)
    },

    smooth = function(window = 11, poly = 3) {
      if (length(self$intensity) >= window && window %% 2 == 1) {
        self$smoothed <- signal::sgolayfilt(self$intensity, p = poly, n = window)
      } else {
        # fallback: no smoothing if params are invalid for short series
        self$smoothed <- self$intensity
      }
      invisible(self)
    },

    fit_gaussian = function() {
      if (is.null(self$smoothed)) stop("Smooth the signal first with $smooth().")

      gaussian_model <- function(x, a, b, c)
        a * exp(-((x - b)^2) / (2 * c^2))

      df <- data.frame(time = self$time, smoothed = self$smoothed)

      start <- list(
        a = max(df$smoothed, na.rm = TRUE),
        b = df$time[which.max(df$smoothed)],
        c = (max(df$time) - min(df$time)) / 10
      )

      fit <- minpack.lm::nlsLM(smoothed ~ gaussian_model(time, a, b, c),
                               data = df, start = start)
      self$fit_params <- coef(fit)
      invisible(self)
    },

    baseline = function(method = c("min", "median")) {
      method <- match.arg(method)
      base <- if (method == "min") min(self$intensity) else median(self$intensity)
      self$baseline_subtracted <- self$intensity - base
      invisible(self)
    },

    auc = function(use = c("raw", "smoothed")) {
      use <- match.arg(use)
      y <- if (use == "smoothed" && !is.null(self$smoothed)) self$smoothed else self$intensity
      pracma::trapz(self$time, y)
    },

    # NOTE: thresholds are in *minutes* (distance/width) and absolute intensity for height
    find_peaks = function(min_height = NULL,
                      # kept for API compatibility; ignored
                      min_distance_min = NULL,
                      min_width_min    = NULL,
                      min_prom_frac    = NULL) {

  if (is.null(self$intensity) || is.null(self$time))
    stop("No data available for peak finding.")

  # 1) Work on smoothed signal (robust to noise); fall back to raw if needed
  y <- as.numeric(self$smoothed %||% self$intensity)
  x <- as.numeric(self$time)
  n <- length(y)
  if (n < 3L) { self$peaks <- NULL; return(invisible(self)) }

  # 2) Absolute height threshold (the app already converts % of max to an absolute)
  ymax <- suppressWarnings(max(y, na.rm = TRUE))
  if (!is.finite(ymax)) { self$peaks <- NULL; return(invisible(self)) }
  min_height_abs <- min_height %||% (0.10 * ymax)

  # 3) Identify local maxima (plateau-safe)
  #    We look for rising -> falling transitions; if flat-top, pick the midpoint.
  dy  <- diff(y)
  sdy <- sign(dy)

  # candidate "turning points" where slope switches from >0 to <=0
  r2f <- which(c(sdy, NA) <= 0 & c(NA, sdy) > 0)
  r2f <- r2f[is.finite(r2f)]

  pick_plateau_mid <- function(idx) {
    L <- idx; R <- idx
    # expand left/right while values equal to the candidate (flat top)
    while (L > 1L && y[L - 1L] == y[idx]) L <- L - 1L
    while (R < n  && y[R + 1L] == y[idx]) R <- R + 1L
    as.integer(round((L + R) / 2))
  }

  cand <- integer(0)
  for (i in r2f) {
    if (i <= 1L || i >= n) next
    ii <- if (y[i] == y[i - 1L] || y[i] == y[i + 1L]) pick_plateau_mid(i) else i
    if (ii <= 1L || ii >= n) next
    # local max, finite, and above height threshold
    if (is.finite(y[ii]) && y[ii] >= y[ii - 1L] && y[ii] > y[ii + 1L] && y[ii] >= min_height_abs) {
      cand <- c(cand, ii)
    }
  }
  cand <- sort(unique(cand))
  if (!length(cand)) { self$peaks <- NULL; return(invisible(self)) }

  # 4) Find surrounding local minima to define integration bounds
  #    falling -> rising transitions; also treat the edges as minima sentinels.
  f2r  <- which(c(sdy, NA) >= 0 & c(NA, sdy) < 0)
  mins <- sort(unique(c(1L, f2r, n)))

  left_min_for  <- function(idx) mins[max(1L, findInterval(idx, mins))]
  right_min_for <- function(idx) {
    pos <- findInterval(idx, mins)
    if (pos + 1L <= length(mins)) mins[pos + 1L] else n
  }

  # 5) Build the peaks data.frame
  df <- data.frame(
    position = as.integer(cand),
    time     = x[cand],
    height   = y[cand],
    stringsAsFactors = FALSE
  )

  df$start <- vapply(df$position, left_min_for,  integer(1))
  df$end   <- vapply(df$position, right_min_for, integer(1))

  # Ensure valid bounds and at least 2 points for integration later
  df$start <- pmax(1L, pmin(df$start, n - 1L))
  df$end   <- pmax(df$start + 1L, pmin(df$end, n))

  # Sort by time for stable labeling
  df <- df[order(df$time), , drop = FALSE]

  self$peaks <- if (nrow(df)) df else NULL
  invisible(self)
},

  # FIXED: Integrate areas between surrounding minima using trapezoidal rule.
  #        Relative area = (area / total_area) * 100.
  integrate_peaks = function() {
    if (is.null(self$peaks) || !nrow(self$peaks)) {
      stop("No peaks detected. Run $find_peaks() first.")
    }

    res <- self$peaks

    res$area <- mapply(function(s, e) {
      # clamp to valid range
      s <- as.integer(max(1, min(s, length(self$time))))
      e <- as.integer(max(1, min(e, length(self$time))))
      # need at least 2 points to integrate
      if ((e - s + 1L) < 2L) return(NA_real_)
      xx <- as.numeric(self$time[s:e])
      yy <- as.numeric(self$intensity[s:e])  # integrate RAW intensity
      if (length(xx) < 2 || anyNA(xx) || anyNA(yy)) return(NA_real_)
      # trapezoidal area
      tryCatch(pracma::trapz(xx, yy), error = function(.) NA_real_)
    }, res$start, res$end)

    # keep only valid, positive areas
    res <- res[is.finite(res$area) & res$area > 0, , drop = FALSE]

    # relative area percentage
    if (nrow(res)) {
      total <- sum(res$area, na.rm = TRUE)
      res$rel_area_pct <- if (is.finite(total) && total > 0) (res$area / total) * 100 else NA_real_
    }

    self$peaks <- res
    invisible(self)
  },

  plot = function(save_path = NULL) {
    df <- data.frame(time = self$time, intensity = self$intensity)
    p <- ggplot2::ggplot(df, ggplot2::aes(x = time, y = intensity)) +
      ggplot2::geom_line(linewidth = 0.9) +
      ggplot2::labs(x = "Time (min)", y = "Intensity") +
      ggplot2::theme_classic(base_size = 13)

    if (!is.null(self$smoothed)) {
      df$smoothed <- self$smoothed
      p <- p + ggplot2::geom_line(data = df, ggplot2::aes(y = smoothed),
                                  linewidth = 0.8, linetype = "dashed", color = "blue")
    }

    if (!is.null(self$fit_params)) {
      gaussian <- function(x, a, b, c) a * exp(-((x - b)^2) / (2 * c^2))
      df$fit <- gaussian(df$time, self$fit_params["a"], self$fit_params["b"], self$fit_params["c"])
      p <- p + ggplot2::geom_line(data = df, ggplot2::aes(y = fit),
                                  linewidth = 0.8, linetype = "dotted", color = "red")
    }

    if (!is.null(self$peaks) && nrow(self$peaks) > 0) {
      p <- p + ggplot2::geom_vline(data = self$peaks, ggplot2::aes(xintercept = time),
                                   linetype = "dotted", linewidth = 0.6, color = "gray40")
    }

    if (!is.null(save_path)) {
      dir.create(dirname(save_path), recursive = TRUE, showWarnings = FALSE)
      ggplot2::ggsave(save_path, plot = p, width = 7, height = 5, dpi = 300)
    }

    p
  },

  summarize_peaks = function() {
    if (is.null(self$peaks) || !nrow(self$peaks)) {
      cat("No peaks detected.\n")
      return(invisible(NULL))
    }
    pk <- self$peaks
    cat(sprintf("Detected %d peak(s):\n", nrow(pk)))
    for (i in seq_len(nrow(pk))) {
      area_str <- if ("rel_area_pct" %in% names(pk) && is.finite(pk$rel_area_pct[i])) {
        sprintf(", Area: %.1f%%", pk$rel_area_pct[i])
      } else {
        ""
      }
      cat(sprintf("  Peak %d: time = %.3f, height = %.1f%s\n", i, pk$time[i], pk$height[i], area_str))
    }
    invisible(pk)
  }

  )
)
