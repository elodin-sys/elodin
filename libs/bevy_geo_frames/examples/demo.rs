use bevy::color::palettes::css;
use bevy::math::primitives::{Cuboid, Plane3d};
use bevy::math::{DQuat, DVec3};
use bevy::prelude::*;
use bevy_editor_cam::prelude::*;
use bevy_geo_frames::*;
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin, InfiniteGridSettings};
use map_3d::Ellipsoid;

/// Marker for the demo cuboid we switch frames on.
#[derive(Component)]
struct FrameDemo;

/// Marker for the infinite grid entity.
#[derive(Component)]
struct GridMarker;

/// Tracks the currently selected frame for gizmo placement.
#[derive(Resource, Clone, Copy)]
struct CurrentFrame {
    frame: GeoFrame,
}

impl Default for CurrentFrame {
    fn default() -> Self {
        Self {
            frame: GeoFrame::ENU,
        }
    }
}

fn main() {
    let mut app = App::new();

    app.add_plugins(DefaultPlugins)
        .add_plugins(DefaultEditorCamPlugins)
        .add_plugins(InfiniteGridPlugin)
        .add_plugins(GeoFramePlugin {
            origin: Some(
                GeoOrigin::new_from_degrees(0.0, 0.0, 0.0)
                    .with_ellipsoid(Ellipsoid::Sphere { radius: 10.0 }),
            ),
            ..default()
        })
        .add_systems(Startup, (setup, setup_ui))
        .add_systems(
            Update,
            (
                frame_switch_input,
                transform_frame_at_position,
                update_position_display,
                toggle_present_mode,
            ),
        )
        .add_systems(Update, draw_origin_gizmos)
        .add_systems(Update, draw_frame_zero_gizmo)
        // .add_systems(Update, draw_frame_axes)
        .add_systems(Update, draw_radius_sphere)
        .init_resource::<CurrentFrame>();

    #[cfg(feature = "big_space")]
    app.add_plugins(::big_space::FloatingOriginPlugin::<i128>::new(
        16_000., 100.,
    ))
    .add_plugins(::big_space::debug::FloatingOriginDebugPlugin::<i128>::default())
    .add_plugins(bevy_geo_frames::big_space::plugin::<i128>);
    app.run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera with editor controls
    let camera_id = commands
        .spawn((
            Camera3d::default(),
            Transform::from_xyz(30.0, 20.0, 30.0).looking_at(Vec3::ZERO, Vec3::Y),
            EditorCam::default(),
        ))
        .id();
    #[cfg(feature = "big_space")]
    commands.entity(camera_id).insert((
        ::big_space::FloatingOrigin,
        ::big_space::GridCell::<i128>::default(),
    ));

    // Light
    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 50.0, 0.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Ground plane for reference (unused, kept for quick toggling).
    let _ground_mesh = meshes.add(Plane3d::default().mesh().size(100.0, 100.0));
    let _ground_mat = materials.add(Color::srgb(0.1, 0.1, 0.1));

    // Infinite grid (visible in Plane mode).
    commands
        .spawn((InfiniteGridBundle::default(), GridMarker))
        .insert(InfiniteGridSettings {
            x_axis_color: css::PINK.into(),
            z_axis_color: css::DARK_CYAN.into(),
            ..default()
        });

    // Demo cuboid
    let cuboid_mesh = meshes.add(Cuboid::new(1.0, 2.0, 3.0));
    let cuboid_mat = materials.add(Color::srgb(0.3, 0.8, 0.9));

    // Position in ENU frame to start: 20 m east, 0 m north, 1 m up
    let enu_pos = DVec3::new(0.0, 0.0, 0.0);

    let cube_id = commands
        .spawn((
            Mesh3d(cuboid_mesh),
            MeshMaterial3d(cuboid_mat),
            Transform::default(),
            GeoPosition(GeoFrame::ENU, enu_pos),
            GeoVelocity(GeoFrame::ENU, DVec3::new(0.1, 0.0, 0.0)),
            GeoRotation(GeoFrame::ENU, DQuat::IDENTITY),
            GeoAngularVelocity(
                GeoFrame::ENU,
                // DVec3::new(0.0, 0.0, 10.0_f32.to_radians()),
                // DVec3::new(10.0, 0.0, 0.0),
                // DVec3::new(0.0, 1.0, 0.0),
                DVec3::new(0.0, 0.0, 1.0),
            ),
            FrameDemo,
        ))
        .id();

    #[cfg(feature = "big_space")]
    commands
        .entity(cube_id)
        .insert((::big_space::GridCell::<i128>::default(),));
}

/// Marker component for the position display text.
#[derive(Component)]
struct PositionDisplay;

fn setup_ui(mut commands: Commands) {
    // Spawn a text entity to display position and frame info
    commands.spawn((
        Text::new(""),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
        PositionDisplay,
    ));
}

fn update_position_display(
    q: Query<(&GeoPosition, &Transform), With<FrameDemo>>,
    mut text_query: Query<&mut Text, With<PositionDisplay>>,
) {
    if let Ok((geo_trans, transform)) = q.single() {
        let frame = geo_trans.0;
        let pos_in_frame = geo_trans.1;
        let pos_in_bevy = transform.translation;

        let frame_name = format!("{:?}", frame);
        let text = format!(
            "Frame: {}\nPosition in frame: ({:.2}, {:.2}, {:.2})\nPosition in Bevy: ({:.2}, {:.2}, {:.2})",
            frame_name,
            pos_in_frame.x,
            pos_in_frame.y,
            pos_in_frame.z,
            pos_in_bevy.x,
            pos_in_bevy.y,
            pos_in_bevy.z
        );

        if let Ok(mut text_component) = text_query.single_mut() {
            *text_component = Text::new(text);
        }
    }
}

/// Toggle between Plane and Sphere presentation mode.
fn toggle_present_mode(
    keys: Res<ButtonInput<KeyCode>>,
    mut ctx: ResMut<GeoContext>,
    mut grid_query: Query<&mut Visibility, With<GridMarker>>,
) {
    if keys.just_pressed(KeyCode::KeyP) {
        ctx.present = match ctx.present {
            Present::Plane => Present::Sphere,
            Present::Sphere => Present::Plane,
        };

        for mut visibility in &mut grid_query {
            *visibility = match ctx.present {
                Present::Plane => Visibility::Visible,
                Present::Sphere => Visibility::Hidden,
            };
        }

        info!(?ctx.present, "Toggled present mode");
    }
}

/// Handle keyboard input to switch the demo cuboid between frames.
///
/// 1: ENU
/// 2: NED
/// 3: ECEF
fn frame_switch_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut q: Query<&mut GeoPosition, With<FrameDemo>>,
    mut current_frame: ResMut<CurrentFrame>,
) {
    let mut target_frame: Option<GeoFrame> = None;

    if keys.just_pressed(KeyCode::Digit1) {
        target_frame = Some(GeoFrame::ENU);
    } else if keys.just_pressed(KeyCode::Digit2) {
        target_frame = Some(GeoFrame::NED);
    } else if keys.just_pressed(KeyCode::Digit3) {
        target_frame = Some(GeoFrame::ECEF);
    }

    if let Some(frame) = target_frame {
        for mut geo in &mut q {
            geo.0 = frame;
            info!(?frame, "Switched demo cuboid to frame");
        }
        current_frame.frame = frame;
    }
}

/// Transform the frame of the object while keeping its Bevy position constant.
///
/// Q: ENU
/// W: NED
/// E: ECEF
fn transform_frame_at_position(
    keys: Res<ButtonInput<KeyCode>>,
    ctx: Res<GeoContext>,
    mut q: Query<(&Transform, &mut GeoPosition), With<FrameDemo>>,
    mut current_frame: ResMut<CurrentFrame>,
) {
    let mut target_frame: Option<GeoFrame> = None;

    if keys.just_pressed(KeyCode::KeyQ) {
        target_frame = Some(GeoFrame::ENU);
    } else if keys.just_pressed(KeyCode::KeyW) {
        target_frame = Some(GeoFrame::NED);
    } else if keys.just_pressed(KeyCode::KeyE) {
        target_frame = Some(GeoFrame::ECEF);
    }

    if let Some(frame) = target_frame {
        for (transform, mut geo_trans) in &mut q {
            // Get current Bevy position
            let bevy_pos = transform.translation;

            // Convert from Bevy to the target frame
            let pos_in_frame = GeoPosition::from_bevy(frame, bevy_pos, &ctx).1;

            // Update the frame and position
            geo_trans.0 = frame;
            geo_trans.1 = pos_in_frame;
            current_frame.frame = frame;

            info!(
                ?frame,
                "Transformed frame to {:?} at position ({:.2}, {:.2}, {:.2})",
                frame,
                pos_in_frame.x,
                pos_in_frame.y,
                pos_in_frame.z
            );
        }
    }
}

/// Draw gizmos at the world origin to show Bevy axes.
fn draw_origin_gizmos(mut gizmos: Gizmos, ctx: Res<GeoContext>, current_frame: Res<CurrentFrame>) {
    let bevy_M_frame = GeoFrame::bevy_M_(&current_frame.frame, &ctx);
    let origin = bevy_M_frame.transform_point3(DVec3::ZERO).as_vec3();
    let l = 5.0;

    // X axis (East)
    gizmos.line(
        origin,
        origin + l * bevy_M_frame.x_axis.xyz().as_vec3(),
        Color::srgb(1.0, 0.0, 0.0),
    );
    // Y axis (Up)
    gizmos.line(
        origin,
        origin + l * bevy_M_frame.y_axis.xyz().as_vec3(),
        Color::srgb(0.0, 1.0, 0.0),
    );
    // Z axis (South)
    gizmos.line(
        origin,
        origin + l * bevy_M_frame.z_axis.xyz().as_vec3(),
        Color::srgb(0.0, 0.0, 1.0),
    );

    // // A little cube-ish marker at the origin
    // gizmos.cuboid(
    //     Transform::from_translation(origin).with_scale(Vec3::splat(0.5)),
    //     Color::srgb(1.0, 1.0, 1.0),
    // );
}

/// Draw RGB axes on the object showing the frame's coordinate directions.
/// Red = first component, Green = second component, Blue = third component.
fn draw_frame_axes(
    mut gizmos: Gizmos,
    ctx: Res<GeoContext>,
    q: Query<(&Transform, &GeoPosition), With<FrameDemo>>,
) {
    let axis_length = 3.0;

    for (transform, geo_trans) in &q {
        let frame = geo_trans.0;

        // Get the basis matrix - columns are the frame's basis vectors in Bevy world space.
        let basis_mat = GeoFrame::bevy_R_(&frame, &ctx);

        // Extract the three basis vectors (columns of the matrix).
        // These represent the first, second, and third component directions.
        // The basis vectors are already in Bevy world space, so we use them directly
        // without applying the object's rotation so the axes stay fixed in world space.
        let axis1 = basis_mat.x_axis.as_vec3(); // First component (Red)
        let axis2 = basis_mat.y_axis.as_vec3(); // Second component (Green)
        let axis3 = basis_mat.z_axis.as_vec3(); // Third component (Blue)

        let pos = transform.translation;

        // Draw lines for each axis in RGB colors
        // Red = first component
        gizmos.line(pos, pos + axis1 * axis_length, Color::srgb(1.0, 0.0, 0.0));
        // Green = second component
        gizmos.line(pos, pos + axis2 * axis_length, Color::srgb(0.0, 1.0, 0.0));
        // Blue = third component
        gizmos.line(pos, pos + axis3 * axis_length, Color::srgb(0.0, 0.0, 1.0));
    }
}

/// Draw a red point for the zero position of the active frame, in the current presentation mode.
fn draw_frame_zero_gizmo(
    mut gizmos: Gizmos,
    ctx: Res<GeoContext>,
    q: Query<&GeoPosition, With<FrameDemo>>,
) {
    for geo_pos in &q {
        let zero = GeoPosition(geo_pos.0, DVec3::ZERO);
        let pos = zero.to_bevy(&ctx);
        gizmos.cuboid(
            Transform::from_translation(pos.as_vec3()).with_scale(Vec3::splat(0.3)),
            Color::srgb(1.0, 0.0, 0.0),
        );
    }
}

/// Draw a wireframe sphere with radius equal to the reference radius from `GeoContext`.
fn draw_radius_sphere(mut gizmos: Gizmos, ctx: Res<GeoContext>) {
    if ctx.present != Present::Sphere {
        return;
    }
    let radius = approx_radius(&ctx.origin.ellipsoid) as f32;
    let center = Vec3::ZERO;

    // Draw wireframe sphere using gizmos
    // Draw multiple circles to create a sphere wireframe
    let segments = 16;
    let color = Color::srgb(0.5, 0.5, 0.5);

    // Draw circles in XY plane (varying Z)
    for i in 0..=segments {
        let z = -radius + (2.0 * radius * i as f32 / segments as f32);
        let circle_radius = (radius * radius - z * z).max(0.0).sqrt();
        if circle_radius > 0.01 {
            // Draw circle in XY plane at this Z
            for j in 0..segments {
                let angle1 = 2.0 * std::f32::consts::PI * j as f32 / segments as f32;
                let angle2 = 2.0 * std::f32::consts::PI * (j + 1) as f32 / segments as f32;
                let p1 = center
                    + Vec3::new(
                        circle_radius * angle1.cos(),
                        circle_radius * angle1.sin(),
                        z,
                    );
                let p2 = center
                    + Vec3::new(
                        circle_radius * angle2.cos(),
                        circle_radius * angle2.sin(),
                        z,
                    );
                gizmos.line(p1, p2, color);
            }
        }
    }

    // Draw circles in XZ plane (varying Y)
    for i in 0..=segments {
        let y = -radius + (2.0 * radius * i as f32 / segments as f32);
        let circle_radius = (radius * radius - y * y).max(0.0).sqrt();
        if circle_radius > 0.01 {
            for j in 0..segments {
                let angle1 = 2.0 * std::f32::consts::PI * j as f32 / segments as f32;
                let angle2 = 2.0 * std::f32::consts::PI * (j + 1) as f32 / segments as f32;
                let p1 = center
                    + Vec3::new(
                        circle_radius * angle1.cos(),
                        y,
                        circle_radius * angle1.sin(),
                    );
                let p2 = center
                    + Vec3::new(
                        circle_radius * angle2.cos(),
                        y,
                        circle_radius * angle2.sin(),
                    );
                gizmos.line(p1, p2, color);
            }
        }
    }

    // Draw circles in YZ plane (varying X)
    for i in 0..=segments {
        let x = -radius + (2.0 * radius * i as f32 / segments as f32);
        let circle_radius = (radius * radius - x * x).max(0.0).sqrt();
        if circle_radius > 0.01 {
            for j in 0..segments {
                let angle1 = 2.0 * std::f32::consts::PI * j as f32 / segments as f32;
                let angle2 = 2.0 * std::f32::consts::PI * (j + 1) as f32 / segments as f32;
                let p1 = center
                    + Vec3::new(
                        x,
                        circle_radius * angle1.cos(),
                        circle_radius * angle1.sin(),
                    );
                let p2 = center
                    + Vec3::new(
                        x,
                        circle_radius * angle2.cos(),
                        circle_radius * angle2.sin(),
                    );
                gizmos.line(p1, p2, color);
            }
        }
    }
}
