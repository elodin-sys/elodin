//! Default placement is computed independently for each physical display.
use std::collections::BTreeMap;

use bevy::{prelude::Entity, winit::WINIT_WINDOWS};
use impeller_wkt::WindowRect;

use super::placement::{collect_sorted_screens, detect_window_screen};
use crate::ui::tiles::WindowDescriptor;

#[derive(bevy::prelude::Component)]
pub struct AutoArranged;

pub fn arrange_windows(
    primary: Entity,
    main: &mut Option<WindowDescriptor>,
    secondary: &mut [WindowDescriptor],
) {
    if secondary.is_empty() && main.as_ref().is_none_or(|d| d.screen.is_none()) {
        return;
    }
    let Some((screens, fallback)) = WINIT_WINDOWS.with_borrow(|windows| {
        let window = windows.get_window(primary)?;
        let screens = collect_sorted_screens(window);
        let fallback = detect_window_screen(window, &screens)?;
        Some((screens, fallback))
    }) else {
        return;
    };
    let main = main.get_or_insert_with(Default::default);
    let primary_screen = main.screen.unwrap_or(fallback);
    let mut groups = BTreeMap::<usize, Vec<usize>>::new();
    groups.entry(primary_screen).or_default();
    for (index, descriptor) in secondary.iter().enumerate() {
        groups
            .entry(descriptor.screen.unwrap_or(fallback))
            .or_default()
            .push(index);
    }
    for (screen_index, indices) in groups {
        let Some(screen) = screens.get(screen_index) else {
            continue;
        };
        let has_primary = screen_index == primary_screen;
        // An explicit rectangle opts this display out of automatic arrangement.
        if (has_primary && main.screen_rect.is_some())
            || indices.iter().any(|&i| secondary[i].screen_rect.is_some())
        {
            continue;
        }
        let size = screen.size();
        let top_height = if has_primary && !indices.is_empty() {
            33
        } else {
            100
        };
        let rects = grid(indices.len(), size.width, size.height, top_height);
        for (index, rect) in indices.into_iter().zip(rects) {
            secondary[index].screen.get_or_insert(screen_index);
            secondary[index].screen_rect = Some(rect);
        }
        if has_primary {
            main.screen.get_or_insert(screen_index);
            main.screen_rect = Some(WindowRect {
                x: 0,
                y: if top_height == 33 { 33 } else { 0 },
                width: 100,
                height: if top_height == 33 { 67 } else { 100 },
            });
        }
    }
}

fn grid(count: usize, width: u32, height: u32, percent_height: u32) -> Vec<WindowRect> {
    if count == 0 {
        return Vec::new();
    }
    let aspect = width.max(1) as f64 / (height.max(1) as f64 * percent_height as f64 / 100.0);
    let columns = (1..=count)
        .min_by(|&a, &b| {
            let score = |columns: usize| {
                let rows = count.div_ceil(columns);
                let mut score = 0.0;
                for row in 0..rows {
                    let cells = columns.min(count - row * columns);
                    score += cells as f64 * (aspect * rows as f64 / cells as f64 / 1.6).ln().abs();
                }
                score
            };
            score(a).total_cmp(&score(b))
        })
        .unwrap();
    let rows = count.div_ceil(columns);
    (0..count)
        .map(|index| {
            let row = index / columns;
            let col = index % columns;
            let cells = columns.min(count - row * columns);
            let x = col * 100 / cells;
            let right = (col + 1) * 100 / cells;
            let y = row * percent_height as usize / rows;
            let bottom = (row + 1) * percent_height as usize / rows;
            WindowRect {
                x: x as u32,
                y: y as u32,
                width: (right - x) as u32,
                height: (bottom - y) as u32,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_secondary_windows_share_the_top_third() {
        assert_eq!(
            grid(2, 1168, 729, 33),
            vec![
                WindowRect {
                    x: 0,
                    y: 0,
                    width: 50,
                    height: 33
                },
                WindowRect {
                    x: 50,
                    y: 0,
                    width: 50,
                    height: 33
                },
            ]
        );
    }

    #[test]
    fn grids_fill_their_region_without_overlap() {
        for count in [1, 2, 3, 11, 12] {
            for height in [33, 50, 100] {
                let rects = grid(count, 1920, 1080, height);
                assert_eq!(rects.len(), count);
                assert_eq!(
                    rects.iter().map(|r| r.width * r.height).sum::<u32>(),
                    100 * height
                );
                for (i, a) in rects.iter().enumerate() {
                    assert!(a.width > 0 && a.height > 0);
                    assert!(a.x + a.width <= 100 && a.y + a.height <= height);
                    for b in &rects[i + 1..] {
                        assert!(
                            a.x + a.width <= b.x
                                || b.x + b.width <= a.x
                                || a.y + a.height <= b.y
                                || b.y + b.height <= a.y
                        );
                    }
                }
            }
        }
    }
}
