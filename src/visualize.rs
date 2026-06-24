//! Animated trajectory visualizer (requires `--features visualize`).
//!
//! Opens an egui window and plays back a recorded `Vec<(f64, Vec<f64>)>`
//! trajectory as an animated line-graph.  Useful for inspecting 1-D PDE
//! solutions (spatial profile evolving in time) or small ODE systems
//! (each component plotted as a separate line).
//!
//! # Example
//! ```no_run
//! use chem_solver_rs::visualize::TrajectoryPlayer;
//!
//! let traj: Vec<(f64, Vec<f64>)> = vec![(0.0, vec![1.0, 0.5]), (0.1, vec![0.9, 0.4])];
//! TrajectoryPlayer::new(traj).play().unwrap();
//! ```

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

/// Plays back an ODE/PDE trajectory as an animated egui window.
///
/// * For **spatial** problems (1-D PDE) the x-axis defaults to the grid index
///   `0..n`.  Supply a custom `x_values` slice to use physical coordinates.
/// * For **multi-component** ODE problems each component is a separate line,
///   labelled `y[0]`, `y[1]`, …
pub struct TrajectoryPlayer {
    frames: Vec<(f64, Vec<f64>)>,
    title: String,
    fps: f64,
    x_label: String,
    y_label: String,
    x_values: Option<Vec<f64>>,
}

impl TrajectoryPlayer {
    /// Create a player from a trajectory produced by
    /// [`BackwardEuler::integrate`](crate::stepper::BackwardEuler::integrate).
    pub fn new(frames: Vec<(f64, Vec<f64>)>) -> Self {
        Self {
            frames,
            title: "Trajectory".to_string(),
            fps: 30.0,
            x_label: "index".to_string(),
            y_label: "value".to_string(),
            x_values: None,
        }
    }

    /// Override the window / plot title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Playback speed in frames per second (default: 30).
    pub fn with_fps(mut self, fps: f64) -> Self {
        self.fps = fps.max(0.1);
        self
    }

    /// Physical x-coordinates for spatial plots.
    ///
    /// Must have length `n` (same as the state vectors in the trajectory).
    pub fn with_x_values(mut self, xs: Vec<f64>) -> Self {
        self.x_values = Some(xs);
        self
    }

    /// Axis labels.
    pub fn with_labels(mut self, x: impl Into<String>, y: impl Into<String>) -> Self {
        self.x_label = x.into();
        self.y_label = y.into();
        self
    }

    /// Open the window and block until the user closes it.
    pub fn play(self) -> Result<(), eframe::Error> {
        let title = self.title.clone();
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_title(&title)
                .with_inner_size([900.0, 600.0]),
            ..Default::default()
        };
        eframe::run_native(
            &title,
            options,
            Box::new(|_cc| Ok(Box::new(PlayerApp::new(self)))),
        )
    }
}

// ── Internal egui App ────────────────────────────────────────────────────────

struct PlayerApp {
    frames: Vec<(f64, Vec<f64>)>,
    x_values: Option<Vec<f64>>,
    x_label: String,
    y_label: String,
    /// Seconds between frames.
    frame_dt: f64,
    /// Current (possibly fractional) frame index.
    cursor: f64,
    playing: bool,
    /// Wall-clock time of last egui repaint (for animation pacing).
    last_tick: Option<std::time::Instant>,
}

impl PlayerApp {
    fn new(player: TrajectoryPlayer) -> Self {
        Self {
            frame_dt: 1.0 / player.fps,
            frames: player.frames,
            x_values: player.x_values,
            x_label: player.x_label,
            y_label: player.y_label,
            cursor: 0.0,
            playing: true,
            last_tick: None,
        }
    }

    fn n_frames(&self) -> usize {
        self.frames.len()
    }

    fn current_frame(&self) -> usize {
        (self.cursor as usize).min(self.n_frames().saturating_sub(1))
    }

    fn advance(&mut self) {
        let now = std::time::Instant::now();
        if let Some(last) = self.last_tick.take() {
            let elapsed = now.duration_since(last).as_secs_f64();
            self.cursor += elapsed / self.frame_dt;
            if self.cursor >= self.n_frames() as f64 {
                self.cursor = 0.0; // loop
            }
        }
        self.last_tick = Some(now);
    }
}

impl eframe::App for PlayerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.playing {
            self.advance();
            ctx.request_repaint();
        }

        let fidx = self.current_frame();
        let n_frames = self.n_frames();
        let (t, y) = {
            let (ft, fy) = &self.frames[fidx];
            (*ft, fy.clone())
        };
        let n = y.len();

        egui::TopBottomPanel::bottom("controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if self.playing {
                    if ui.button("⏸ Pause").clicked() {
                        self.playing = false;
                        self.last_tick = None;
                    }
                } else if ui.button("▶ Play").clicked() {
                    self.playing = true;
                }

                ui.separator();

                let mut fidx_mut = fidx;
                let resp = ui.add(
                    egui::Slider::new(&mut fidx_mut, 0..=n_frames.saturating_sub(1))
                        .text("frame"),
                );
                if resp.changed() {
                    self.cursor = fidx_mut as f64;
                    self.playing = false;
                    self.last_tick = None;
                }

                ui.separator();
                ui.label(format!(
                    "t = {t:.4e}   frame {fidx}/{tot}",
                    tot = n_frames.saturating_sub(1)
                ));

                ui.separator();
                let mut fps = 1.0 / self.frame_dt;
                if ui
                    .add(egui::Slider::new(&mut fps, 1.0..=120.0).text("fps"))
                    .changed()
                {
                    self.frame_dt = 1.0 / fps;
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            Plot::new("trajectory")
                .x_axis_label(self.x_label.clone())
                .y_axis_label(self.y_label.clone())
                .show(ui, |plot_ui| {
                    if n <= 1 {
                        // Single-component: show time-series up to current frame.
                        let pts: PlotPoints = self.frames[..=fidx]
                            .iter()
                            .map(|(ft, fy)| [*ft, fy[0]])
                            .collect();
                        plot_ui.line(Line::new("y[0]", pts));
                    } else if let Some(xs) = &self.x_values {
                        // Spatial profile with physical x.
                        let pts: PlotPoints =
                            xs.iter().zip(y.iter()).map(|(&x, &v)| [x, v]).collect();
                        plot_ui.line(Line::new(format!("t={t:.4e}"), pts));
                    } else {
                        // Spatial profile with index x, or multi-component ODE.
                        // Heuristic: if dim ≥ 4 treat as spatial, else as components.
                        if n >= 4 {
                            let pts: PlotPoints = (0..n)
                                .map(|i| [i as f64, y[i]])
                                .collect();
                            plot_ui.line(Line::new(format!("t={t:.4e}"), pts));
                        } else {
                            // Multi-component: each component as its own time-series.
                            for comp in 0..n {
                                let pts: PlotPoints = self.frames[..=fidx]
                                    .iter()
                                    .map(|(ft, fy)| [*ft, fy[comp]])
                                    .collect();
                                plot_ui.line(Line::new(format!("y[{comp}]"), pts));
                            }
                        }
                    }
                });
        });
    }
}
