# Chromatography Peak Analysis App

An interactive **R Shiny application** for visualizing, detecting, and integrating peaks in chromatographic data.  
Upload one or multiple CSV files containing `(time, intensity)` pairs and the app performs smoothing, peak detection, trapezoidal integration, and percentage area calculation.

---

## Features

- **Peak detection**
  - Height-only threshold (% of max signal)
  - Savitzky–Golay smoothing for noise reduction
  - Robust trapezoidal peak integration (pracma::trapz)
  - Automated relative area (%) calculation

- **Visualization**
  - Overlay or separate-panel modes
  - Automatic, balanced viridis color palette
  - Optional base color override
  - Peak labels with optional % areas

- **Export**
  - Summary CSV with peak times and relative areas
  - Plot export as **PNG**, **TIFF**, or **EPS**

- **Demo data** included for quick testing

---

## Installation

### Clone the repository
```bash
git clone https://github.com/andrewvgrassetti/Chromatography.git
cd Chromatography
```

### Install Dependencies

Open R and run:

```r
install.packages(c(
  "shiny","bslib","readr","ggplot2","colourpicker","scales",
  "dplyr","purrr","tidyr","stringr","signal","minpack.lm","pracma","viridisLite"
))
# Optional: high-quality graphics
install.packages(c("ragg","Cairo"))
```

Alternatively, if using renv:

```r
install.packages("renv")
renv::restore()
```

### Running the App

```r
library(shiny)
shiny::runApp(".")
```

A browser window will open automatically.
If not, copy the printed URL (e.g., http://127.0.0.1:7428) into your browser.

## Input Format

Each chromatogram file must be a 2-column CSV:

| time | intensity |
|------|-----------|
| 0.00 | 12.3     |
| 0.01 | 12.9     |
| ...  | ...      |

- Time in minutes (or any consistent unit)
- Intensity in arbitrary units
- Multiple files may be uploaded simultaneously.

## Output

### Summary CSV

One row per file including:

- total number of peaks
- per-peak times
- per-peak relative area (%)

### Plot export

Available formats:

- PNG
- TIFF
- EPS

with adjustable size and DPI.

## Testing

Run automated tests:

```bash
Rscript scripts/test_chrom.R
```

Expected output includes checks for:

- peak count accuracy
- integration correctness
- stability with noise

## Project Structure

```
Chromatography/
├── app.R                  # Main Shiny application
├── R/
│   └── Chromatogram.R     # R6 class for smoothing, peak detection, integration
├── scripts/
│   ├── test_chrom.R       # Automated testing for integration & detection
│   └── setup_environment.R # Environment setup script
├── renv.lock              # Package version snapshot (optional)
└── README.md
```
