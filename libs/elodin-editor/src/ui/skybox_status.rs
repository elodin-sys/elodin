use bevy_ai_skybox::prelude::{SkyboxCacheHealth, SkyboxGenerationPhase, SkyboxGenerationUi};
use egui::{self, Label, RichText, Spinner};

use crate::ui::colors::{get_scheme, with_opacity};

const MODAL_WIDTH: f32 = 400.0;
const PROMPT_MAX_CHARS: usize = 72;

pub fn draw_skybox_generation_overlay(ctx: &egui::Context, skybox: &SkyboxGenerationUi) {
    if skybox.is_busy() {
        draw_busy_modal(ctx, skybox);
    }
}

pub fn draw_skybox_status_bar(
    ui: &mut egui::Ui,
    skybox: &SkyboxGenerationUi,
    cache_health: &SkyboxCacheHealth,
) {
    if let Some(error) = &cache_health.load_error {
        let scheme = get_scheme();
        ui.label(
            RichText::new(format!("Skybox cache unavailable: {error}"))
                .text_style(egui::TextStyle::Small)
                .color(scheme.error),
        );
        return;
    }

    if skybox.phase == SkyboxGenerationPhase::Idle {
        return;
    }

    let scheme = get_scheme();
    let message = skybox.message.clone().unwrap_or_else(|| "Skybox".into());

    match skybox.phase {
        SkyboxGenerationPhase::Generating | SkyboxGenerationPhase::PendingApply => {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;
                ui.add(Spinner::new().size(14.0).color(scheme.text_secondary));
                ui.label(
                    RichText::new(message)
                        .text_style(egui::TextStyle::Small)
                        .color(scheme.text_secondary),
                );
            });
        }
        SkyboxGenerationPhase::Ready => {
            ui.label(
                RichText::new(message)
                    .text_style(egui::TextStyle::Small)
                    .color(scheme.success),
            );
        }
        SkyboxGenerationPhase::Failed => {
            ui.label(
                RichText::new(message)
                    .text_style(egui::TextStyle::Small)
                    .color(scheme.error),
            );
        }
        SkyboxGenerationPhase::Idle => {}
    }
}

fn draw_busy_modal(ctx: &egui::Context, skybox: &SkyboxGenerationUi) {
    let scheme = get_scheme();
    let (title, subtitle) = busy_modal_copy(skybox);
    let modal_id = egui::Id::new("skybox_generation_modal");
    let modal_size = egui::vec2(MODAL_WIDTH, if subtitle.is_empty() { 72.0 } else { 96.0 });
    let modal_rect = egui::Rect::from_center_size(ctx.content_rect().center(), modal_size);

    egui::Modal::new(modal_id)
        .area(
            egui::Area::new(modal_id)
                .kind(egui::UiKind::Modal)
                .fixed_pos(modal_rect.min)
                .order(egui::Order::Foreground),
        )
        .backdrop_color(egui::Color32::from_black_alpha(120))
        .frame(egui::Frame {
            fill: with_opacity(scheme.bg_secondary, 0.98),
            stroke: egui::Stroke::new(1.0, with_opacity(scheme.border_primary, 0.6)),
            inner_margin: egui::Margin::symmetric(28, 20),
            corner_radius: egui::CornerRadius::same(12),
            ..Default::default()
        })
        .show(ctx, |ui| {
            ui.set_min_size(modal_size);
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 20.0;
                ui.add(Spinner::new().size(28.0).color(scheme.text_primary));
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing.y = 4.0;
                    ui.label(
                        RichText::new(title)
                            .size(15.0)
                            .strong()
                            .color(scheme.text_primary),
                    );
                    if !subtitle.is_empty() {
                        ui.add(
                            Label::new(
                                RichText::new(subtitle)
                                    .size(12.0)
                                    .color(scheme.text_tertiary),
                            )
                            .wrap_mode(egui::TextWrapMode::Truncate),
                        );
                    }
                });
            });
        });
}

fn busy_modal_copy(skybox: &SkyboxGenerationUi) -> (String, String) {
    match skybox.phase {
        SkyboxGenerationPhase::Generating => {
            let subtitle = skybox
                .prompt
                .as_deref()
                .map(truncate_prompt)
                .unwrap_or_default();
            ("Generating skybox".into(), subtitle)
        }
        SkyboxGenerationPhase::PendingApply => {
            let subtitle = skybox
                .target_name
                .as_deref()
                .map(|name| format!("Loading `{name}` onto viewports"))
                .unwrap_or_else(|| skybox.message.clone().unwrap_or_default());
            ("Applying skybox".into(), subtitle)
        }
        _ => (String::new(), String::new()),
    }
}

fn truncate_prompt(prompt: &str) -> String {
    if prompt.chars().count() <= PROMPT_MAX_CHARS {
        return prompt.to_string();
    }
    let mut end = PROMPT_MAX_CHARS;
    while end > 0 && !prompt.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &prompt[..end])
}
