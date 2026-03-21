/**
 * FootyPredict — Chart.js configuration helpers
 *
 * Exposes a global `FootyPredict` namespace with helper functions for
 * rendering position distribution charts.
 *
 * Chart.js must be loaded before this script.
 */

(function (global) {
  "use strict";

  /**
   * Default Chart.js plugin options shared across all charts.
   */
  const SHARED_PLUGINS = {
    legend: {
      display: false, // Hidden by default — too many positions to label
    },
    tooltip: {
      callbacks: {
        /**
         * Format tooltip title as the team name.
         * @param {import("chart.js").TooltipItem[]} items
         * @returns {string}
         */
        title(items) {
          return items[0]?.label ?? "";
        },
        /**
         * Format tooltip body as "Position N: XX%"
         * @param {import("chart.js").TooltipItem} item
         * @returns {string}
         */
        label(item) {
          const pos = item.datasetIndex + 1;
          const suffix = pos === 1 ? "st" : pos === 2 ? "nd" : pos === 3 ? "rd" : "th";
          const pct = Number(item.parsed.y).toFixed(1);
          return `${pos}${suffix}: ${pct}%`;
        },
      },
    },
  };

  /**
   * Render a stacked bar chart showing position probability distributions.
   *
   * @param {HTMLCanvasElement} canvas - Target canvas element.
   * @param {string[]} teamLabels - Team names (x-axis labels).
   * @param {Array<{label: string, data: number[], backgroundColor: string}>} datasets
   *   One dataset per finishing position.
   * @returns {import("chart.js").Chart} The created Chart.js instance.
   */
  function renderPositionChart(canvas, teamLabels, datasets) {
    if (typeof Chart === "undefined") {
      console.error("FootyPredict: Chart.js is not loaded.");
      return null;
    }

    return new Chart(canvas, {
      type: "bar",
      data: {
        labels: teamLabels,
        datasets: datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            stacked: true,
            ticks: {
              maxRotation: 45,
              minRotation: 30,
              font: { size: 11 },
            },
          },
          y: {
            stacked: true,
            min: 0,
            max: 100,
            ticks: {
              callback: (value) => `${value}%`,
            },
            title: {
              display: true,
              text: "Probability (%)",
            },
          },
        },
        plugins: SHARED_PLUGINS,
        animation: {
          duration: 600,
          easing: "easeOutQuart",
        },
      },
    });
  }

  /**
   * Render a horizontal bar chart for a single team's position distribution.
   *
   * @param {HTMLCanvasElement} canvas - Target canvas element.
   * @param {number[]} posProbs - Array of probabilities indexed by position-1.
   * @param {string} teamName - Team name for the chart title.
   * @returns {import("chart.js").Chart} The created Chart.js instance.
   */
  function renderSingleTeamChart(canvas, posProbs, teamName) {
    if (typeof Chart === "undefined") {
      console.error("FootyPredict: Chart.js is not loaded.");
      return null;
    }

    const labels = posProbs.map((_, i) => {
      const pos = i + 1;
      return `${pos}${pos === 1 ? "st" : pos === 2 ? "nd" : pos === 3 ? "rd" : "th"}`;
    });

    const backgroundColors = posProbs.map((prob) => {
      // Colour scale: blue (low prob) -> red (high prob)
      const intensity = Math.min(1, prob * 5);
      const r = Math.round(21 + 180 * intensity);
      const g = Math.round(101 * (1 - intensity));
      const b = Math.round(192 * (1 - intensity));
      return `rgba(${r}, ${g}, ${b}, 0.8)`;
    });

    return new Chart(canvas, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: teamName,
            data: posProbs.map((p) => +(p * 100).toFixed(2)),
            backgroundColor: backgroundColors,
            borderWidth: 0,
          },
        ],
      },
      options: {
        indexAxis: "y", // Horizontal bars
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            min: 0,
            max: 100,
            ticks: { callback: (v) => `${v}%` },
          },
          y: {
            ticks: { font: { size: 11 } },
          },
        },
        plugins: {
          legend: { display: false },
          title: {
            display: true,
            text: `${teamName} — Position Distribution`,
          },
          tooltip: {
            callbacks: {
              label: (item) => `${Number(item.parsed.x).toFixed(1)}%`,
            },
          },
        },
      },
    });
  }

  /**
   * Known club colours keyed by team name (case-sensitive).
   * Falls back to a generated palette for unknown teams.
   */
  const CLUB_COLOURS = {
    // Premier League
    "Arsenal FC": "#EF0107",
    "Aston Villa FC": "#670E36",
    "AFC Bournemouth": "#DA291C",
    "Brentford FC": "#e30613",
    "Brighton & Hove Albion FC": "#0057B8",
    "Burnley FC": "#6C1D45",
    "Chelsea FC": "#034694",
    "Crystal Palace FC": "#1B458F",
    "Everton FC": "#003399",
    "Fulham FC": "#000000",
    "Leeds United FC": "#FFCD00",
    "Liverpool FC": "#C8102E",
    "Manchester City FC": "#6CABDD",
    "Manchester United FC": "#DA291C",
    "Newcastle United FC": "#241F20",
    "Nottingham Forest FC": "#DD0000",
    "Sheffield United FC": "#EE2737",
    "Sunderland AFC": "#EB172B",
    "Tottenham Hotspur FC": "#132257",
    "West Ham United FC": "#7A263A",
    "Wolverhampton Wanderers FC": "#FDB913",
    // Championship extras
    "Leicester City FC": "#003090",
    "Ipswich Town FC": "#0044AA",
    "Southampton FC": "#D71920",
    "Luton Town FC": "#F78F1E",
  };

  /**
   * Line dash patterns cycled across datasets for additional distinction.
   */
  const DASH_PATTERNS = [
    [],         // solid
    [8, 4],     // dashed
    [2, 4],     // dotted
    [8, 4, 2, 4], // dash-dot
  ];

  /**
   * Generate N visually distinct colours using evenly spaced hues.
   * Used as fallback for teams not in CLUB_COLOURS.
   *
   * @param {number} n - Number of colours needed.
   * @param {number} [alpha=1] - Opacity (0–1).
   * @returns {string[]} Array of HSLA colour strings.
   */
  function _teamPalette(n, alpha) {
    if (alpha === undefined) alpha = 1;
    return Array.from({ length: n }, (_, i) => {
      const hue = (i * 360) / n;
      return `hsla(${hue}, 70%, 50%, ${alpha})`;
    });
  }

  /**
   * Pre-load an image and return a promise that resolves to the Image
   * or null on failure.
   */
  function _loadImage(url, size) {
    return new Promise((resolve) => {
      const img = new Image(size, size);
      img.onload = () => resolve(img);
      img.onerror = () => resolve(null);
      img.src = url;
    });
  }

  /**
   * Render a multi-team prediction timeline (line chart).
   *
   * X-axis = date labels, Y-axis = predicted position (1 at top).
   * One line per team, with clickable legend to toggle visibility.
   * Uses club colours, varied dash patterns, and badge images at data points.
   *
   * @param {HTMLCanvasElement} canvas - Target canvas element.
   * @param {string[]} dates - Date labels for the x-axis.
   * @param {Object<string, number[]>} teamsData - { teamName: [pos, pos, ...] }
   * @param {number} nTeams - Total teams in the league (for y-axis max).
   * @param {Object<string, string>} [crests] - { teamName: crestUrl }
   * @returns {import("chart.js").Chart} The created Chart.js instance.
   */
  function renderTimelineChart(canvas, dates, teamsData, nTeams, crests) {
    if (typeof Chart === "undefined") {
      console.error("FootyPredict: Chart.js is not loaded.");
      return null;
    }

    const BADGE_SIZE = 18;
    const teamNames = Object.keys(teamsData);
    const fallbackColours = _teamPalette(teamNames.length);

    // Pre-load all crest images before creating the chart
    const imagePromises = teamNames.map((name) => {
      const url = crests && crests[name];
      return url ? _loadImage(url, BADGE_SIZE) : Promise.resolve(null);
    });

    // Build chart once images are ready (or immediately if no crests)
    Promise.all(imagePromises).then((images) => {
      const datasets = teamNames.map((name, i) => {
        const colour = CLUB_COLOURS[name] || fallbackColours[i];
        const img = images[i];
        return {
          label: name,
          data: teamsData[name],
          borderColor: colour,
          backgroundColor: colour,
          borderWidth: 2.5,
          borderDash: DASH_PATTERNS[i % DASH_PATTERNS.length],
          pointStyle: img || "circle",
          pointRadius: img ? BADGE_SIZE / 2 : 3,
          pointHoverRadius: img ? BADGE_SIZE / 2 + 4 : 6,
          tension: 0.3,
          fill: false,
        };
      });

      // Helper: make a colour semi-transparent for fading
      function _fadeColour(colour, alpha) {
        if (colour.startsWith("#")) {
          var hex = colour.slice(1);
          if (hex.length === 3) hex = hex[0]+hex[0]+hex[1]+hex[1]+hex[2]+hex[2];
          if (hex.length >= 6) hex = hex.slice(0, 6);
          var a = Math.round(alpha * 255).toString(16).padStart(2, "0");
          return "#" + hex + a;
        }
        if (colour.startsWith("hsl")) {
          var parts = colour.match(/[\d.]+/g);
          return "hsla(" + parts[0] + "," + parts[1] + "%," + parts[2] + "%," + alpha + ")";
        }
        return colour;
      }

      // Save original styles for highlight/restore
      var _origStyles = datasets.map((ds) => ({
        borderColor: ds.borderColor,
        borderWidth: ds.borderWidth,
        pointRadius: ds.pointRadius,
      }));
      var _highlightActive = false;

      function _highlightDataset(chart, hoveredIdx) {
        _highlightActive = true;
        chart.data.datasets.forEach((ds, i) => {
          if (i === hoveredIdx) {
            ds.borderColor = _origStyles[i].borderColor;
            ds.borderWidth = 4.5;
            ds.pointRadius = _origStyles[i].pointRadius;
          } else {
            ds.borderColor = _fadeColour(_origStyles[i].borderColor, 0.12);
            ds.borderWidth = 1;
            ds.pointRadius = 0;
          }
        });
        chart.update("none");
      }

      function _restoreDatasets(chart) {
        if (!_highlightActive) return;
        _highlightActive = false;
        chart.data.datasets.forEach((ds, i) => {
          ds.borderColor = _origStyles[i].borderColor;
          ds.borderWidth = _origStyles[i].borderWidth;
          ds.pointRadius = _origStyles[i].pointRadius;
        });
        chart.update("none");
      }

      // Plugin to restore styles when the mouse leaves the canvas
      const resetOnLeave = {
        id: "resetOnLeave",
        afterEvent(chart, args) {
          if (args.event.type === "mouseout") {
            _restoreDatasets(chart);
          }
        },
      };

      // Belt-and-suspenders: also listen on the DOM element directly
      canvas.addEventListener("mouseleave", () => {
        if (_chart) _restoreDatasets(_chart);
      });

      var _chart = new Chart(canvas, {
        type: "line",
        data: { labels: dates, datasets },
        plugins: [resetOnLeave],
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: {
            mode: "dataset",
            intersect: false,
          },
          onHover(e, elements, chart) {
            if (elements.length > 0) {
              _highlightDataset(chart, elements[0].datasetIndex);
              canvas.style.cursor = "pointer";
            } else {
              _restoreDatasets(chart);
              canvas.style.cursor = "default";
            }
          },
          scales: {
            y: {
              reverse: true,
              min: 1,
              max: nTeams,
              title: { display: true, text: "Predicted Position" },
              ticks: { stepSize: 1, font: { size: 11 } },
            },
            x: {
              ticks: { maxRotation: 45, minRotation: 30, font: { size: 11 } },
            },
          },
          plugins: {
            legend: {
              display: true,
              position: "bottom",
              labels: {
                boxWidth: 12,
                font: { size: 11 },
                usePointStyle: true,
              },
              onHover(e, legendItem, legend) {
                _highlightDataset(legend.chart, legendItem.datasetIndex);
              },
              onLeave(e, legendItem, legend) {
                _restoreDatasets(legend.chart);
              },
            },
            tooltip: {
              callbacks: {
                label: (item) =>
                  `${item.dataset.label}: ${Number(item.parsed.y).toFixed(1)}`,
              },
            },
          },
          animation: { duration: 600, easing: "easeOutQuart" },
        },
      });
    });
  }

  /**
   * Render a single-team prediction timeline (line chart with fill).
   *
   * @param {HTMLCanvasElement} canvas - Target canvas element.
   * @param {string[]} dates - Date labels for the x-axis.
   * @param {number[]} positions - Predicted position over time.
   * @param {string} teamName - Team name for the chart title.
   * @param {number} nTeams - Total teams for y-axis max.
   * @returns {import("chart.js").Chart} The created Chart.js instance.
   */
  function renderTeamTimelineChart(canvas, dates, positions, teamName, nTeams) {
    if (typeof Chart === "undefined") {
      console.error("FootyPredict: Chart.js is not loaded.");
      return null;
    }

    return new Chart(canvas, {
      type: "line",
      data: {
        labels: dates,
        datasets: [
          {
            label: teamName,
            data: positions,
            borderColor: "hsla(210, 70%, 50%, 1)",
            backgroundColor: "hsla(210, 70%, 50%, 0.15)",
            borderWidth: 2.5,
            pointRadius: 3,
            pointHoverRadius: 6,
            tension: 0.3,
            fill: true,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            reverse: true,
            min: 1,
            max: nTeams,
            title: { display: true, text: "Predicted Position" },
            ticks: { stepSize: 1, font: { size: 11 } },
          },
          x: {
            ticks: { maxRotation: 45, minRotation: 30, font: { size: 11 } },
          },
        },
        plugins: {
          legend: { display: false },
          title: { display: true, text: `${teamName} — Position Over Time` },
          tooltip: {
            callbacks: {
              label: (item) => `Position: ${Number(item.parsed.y).toFixed(1)}`,
            },
          },
        },
        animation: { duration: 600, easing: "easeOutQuart" },
      },
    });
  }

  // Expose public API
  global.FootyPredict = {
    renderPositionChart,
    renderSingleTeamChart,
    renderTimelineChart,
    renderTeamTimelineChart,
  };
})(window);
