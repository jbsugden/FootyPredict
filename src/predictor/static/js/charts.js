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
   * Generate N visually distinct colours using evenly spaced hues.
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
   * Render a multi-team prediction timeline (line chart).
   *
   * X-axis = date labels, Y-axis = predicted position (1 at top).
   * One line per team, with clickable legend to toggle visibility.
   *
   * @param {HTMLCanvasElement} canvas - Target canvas element.
   * @param {string[]} dates - Date labels for the x-axis.
   * @param {Object<string, number[]>} teamsData - { teamName: [pos, pos, ...] }
   * @param {number} nTeams - Total teams in the league (for y-axis max).
   * @returns {import("chart.js").Chart} The created Chart.js instance.
   */
  function renderTimelineChart(canvas, dates, teamsData, nTeams) {
    if (typeof Chart === "undefined") {
      console.error("FootyPredict: Chart.js is not loaded.");
      return null;
    }

    const teamNames = Object.keys(teamsData);
    const colours = _teamPalette(teamNames.length);

    const datasets = teamNames.map((name, i) => ({
      label: name,
      data: teamsData[name],
      borderColor: colours[i],
      backgroundColor: colours[i],
      borderWidth: 2,
      pointRadius: 2,
      pointHoverRadius: 5,
      tension: 0.3,
      fill: false,
    }));

    return new Chart(canvas, {
      type: "line",
      data: { labels: dates, datasets },
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
          legend: {
            display: true,
            position: "bottom",
            labels: { boxWidth: 12, font: { size: 11 } },
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
