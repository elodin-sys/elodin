#import bevy_render::view::View

@group(0) @binding(0) var<uniform> view : View;


struct LineUniform {
  line_width : f32,
               color : vec4<f32>,
                       chunk_size : f32,
#ifdef SIXTEEN_BYTE_ALIGNMENT
                                    // WebGL2 structs must be 16 byte aligned.
                                    _padding : vec2<f32>,
#endif
}

@group(1) @binding(0) var<uniform> line_uniform : LineUniform;

@group(2) @binding(0) var<storage> x_values: array<f32>;
@group(2) @binding(1) var<storage> y_values: array<f32>;
@group(2) @binding(2) var<storage> index_buffer: array<u32>;


struct VertexInput {
  @builtin(vertex_index) vertex_index : u32,
  @builtin(instance_index) instance_index : u32,
};

struct VertexOutput {
  @builtin(position) clip_position : vec4<f32>, @location(0) color : vec4<f32>,
};

@vertex fn vertex(vertex : VertexInput) -> VertexOutput {
  // initially based on
  // https://github.com/wwwtyro/instanced-lines-demos/blob/c1e57960b39cf1acce9bd27ded164e5e013fdb36/src/commands.js#L768
  // This implementation has diverged heavily - In particular we are using
  // triangle line strip vertices and many extraneous math operations were
  // removed
  var positions = array<vec2<f32>, 4>(vec2(-1.0, 0.), // bottom-left
                                      vec2(-1.0, 1.), // top-left
                                      vec2(1.0, 0.),  // bottom-right
                                      vec2(1.0, 1.)); // top-right
  let position = positions[vertex.vertex_index];
  let resolution = view.viewport.zw;
  let width = line_uniform.line_width / resolution;
  let index_a = index_buffer[vertex.instance_index];
  let index_b = index_buffer[vertex.instance_index + 1];
  let time_a = x_values[index_a];
  let time_b = x_values[index_b];
  let data_a = y_values[index_a];
  let data_b = y_values[index_b];

  let pos_a = vec2(time_a, data_a);
  let pos_b = vec2(time_b, data_b);
  let clip_a = (view.clip_from_view * vec4(pos_a, 0.0, 1.0)).xy;
  let clip_b = (view.clip_from_view * vec4(pos_b, 0.0, 1.0)).xy;
  let x_basis = normalize(clip_b - clip_a);
  let y_basis = vec2(-x_basis.y, x_basis.x);
  let stride = width * position.x * y_basis;
  let point = mix(clip_a, clip_b, position.y) + stride;
  let clip_position = vec4(point, 0, 1.0);
  return VertexOutput(clip_position, line_uniform.color);
}

struct FragmentInput {
  @builtin(position) position : vec4<f32>, @location(0) color : vec4<f32>,
};

struct FragmentOutput {
  @location(0) color : vec4<f32>,
};

@fragment fn fragment(in : FragmentInput) -> FragmentOutput {
  return FragmentOutput(in.color);
}
