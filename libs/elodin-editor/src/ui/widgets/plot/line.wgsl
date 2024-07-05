#import bevy_render::view::View

@group(0) @binding(0) var<uniform> view: View;


struct LineUniform {
    line_width: f32,
    color: vec4<f32>,
    chunk_size: f32,
#ifdef SIXTEEN_BYTE_ALIGNMENT
    // WebGL2 structs must be 16 byte aligned.
    //_padding: vec3<f32>,
#endif
}

@group(1) @binding(0) var<uniform> line_uniform: LineUniform;

struct VertexInput {
    @location(0) position_a: f32,
    @location(1) position_b: f32,
    @builtin(vertex_index) index: u32,
    @builtin(instance_index) tick: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

const EPSILON: f32 = 4.88e-04;

@vertex
fn vertex(vertex: VertexInput) -> VertexOutput {
    // source: https://github.com/wwwtyro/instanced-lines-demos/blob/c1e57960b39cf1acce9bd27ded164e5e013fdb36/src/commands.js#L768
    var positions = array<vec2<f32>, 6>(
        vec2(-0.5, 0.),
        vec2(-0.5, 1.),
        vec2(0.5, 1.),
        vec2(-0.5, 0.),
        vec2(0.5, 1.),
        vec2(0.5, 0.)
    );
    let width = line_uniform.line_width;
    let position = positions[vertex.index];
    let resolution = view.viewport.zw;
    let x_a = f32(vertex.tick) * line_uniform.chunk_size;
    let pos_a = vec2(x_a, vertex.position_a);
    let pos_b = vec2(x_a + line_uniform.chunk_size, vertex.position_b);
    let clip_a = (view.clip_from_view * vec4(pos_a, 0.0, 1.0)).xy;
    let clip_b = (view.clip_from_view * vec4(pos_b, 0.0, 1.0)).xy;
    let offset_a = resolution * (0.5 * clip_a + 0.5);
    let offset_b = resolution * (0.5 * clip_b + 0.5);
    let x_basis = normalize(offset_b - offset_a);
    let y_basis = vec2(-x_basis.y, x_basis.x);
    let point_a = offset_a + width * (position.x * y_basis);
    let point_b = offset_b + width * (position.x * y_basis);
    let point = mix(point_a, point_b, position.y);
    let clip_position = vec4(2.0 * point / resolution - 1.0, 0, 1.0);
    return VertexOutput(clip_position, line_uniform.color);
}

struct FragmentInput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

struct FragmentOutput {
    @location(0) color: vec4<f32>,
};

@fragment
fn fragment(in: FragmentInput) -> FragmentOutput {
    return FragmentOutput(in.color);
}
