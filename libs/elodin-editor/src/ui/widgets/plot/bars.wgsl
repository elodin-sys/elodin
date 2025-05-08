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

@group(2) @binding(0) var<storage> x_values : array<f32>;
@group(2) @binding(1) var<storage> y_values : array<f32>;
@group(2) @binding(2) var<storage> index_buffer : array<u32>;

struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
                                          @builtin(instance_index) instance_index : u32,
};

struct VertexOutput {
    @builtin(position) clip_position : vec4<f32>,
                                       @location(0) color : vec4<f32>,
};

@vertex fn vertex(vertex : VertexInput) -> VertexOutput
{
    var positions = array<vec2<f32>, 4>(
        vec2(0.0, 0.0), // bottom-left
        vec2(0.0, 1.0), // top-left
        vec2(1.0, 0.0), // bottom-right
        vec2(1.0, 1.0) // top-right
    );

    let position = positions[vertex.vertex_index];
    let resolution = view.viewport.zw;
    let bar_width = line_uniform.line_width / resolution.x;

    let index = index_buffer[vertex.instance_index];
    let time = x_values[index];
    let data = y_values[index];

    let zero_point = (view.clip_from_view * vec4(time, 0.0, 0.0, 1.0)).xy;
    let data_point = (view.clip_from_view * vec4(time, data, 0.0, 1.0)).xy;

    let bar_half_width = bar_width * 4.0;
    let bar_left = zero_point.x - bar_half_width;
    let bar_right = zero_point.x + bar_half_width;

    let x = mix(bar_left, bar_right, position.x);
    let y = mix(zero_point.y, data_point.y, position.y);

    let clip_position = vec4(x, y, 0.0, 1.0);
    return VertexOutput(clip_position, line_uniform.color);
}

struct FragmentInput {
    @builtin(position) position : vec4<f32>,
              @location(0) color : vec4<f32>,
};

struct FragmentOutput {
    @location(0) color : vec4<f32>,
};

@fragment fn fragment(in : FragmentInput) -> FragmentOutput
{
    return FragmentOutput(in.color);
}
