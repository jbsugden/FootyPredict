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

  // Expose public API
  global.FootyPredict = {
    renderPositionChart,
    renderSingleTeamChart,
  };
})(window);
