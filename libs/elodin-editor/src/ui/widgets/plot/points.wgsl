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

    @builtin(position) position : vec4<f32>,
                                  @location(0) point : vec2<f32>,
                                                       @location(1) center : vec2<f32>,
                                                                             @location(2) width : vec2<f32>,
                                                                                                  @location(3) color : vec4<f32>,
};

@vertex fn vertex(vertex : VertexInput) -> VertexOutput
{
    var positions = array<vec2<f32>, 4>(vec2(-1.0, -1.0), // bottom-left
        vec2(-1.0, 1.), // top-left
        vec2(1.0, -1.0), // bottom-right
        vec2(1.0, 1.)); // top-right
    let position = positions[vertex.vertex_index];
    let resolution = view.viewport.zw;
    let width = line_uniform.line_width / resolution;
    let index = index_buffer[vertex.instance_index];
    let time = x_values[index];
    let data = y_values[index];

    let pos = vec2(time, data);
    let clip = (view.clip_from_view * vec4(pos, 0.0, 1.0)).xy;
    let point = position * width * 3.0 + clip;
    let clip_position = vec4(point, 0, 1.0);
    return VertexOutput(clip_position, point, clip, width, line_uniform.color);
}

struct FragmentOutput {
    @location(0) color : vec4<f32>,
};

@fragment fn fragment(in : VertexOutput) -> FragmentOutput
{
    let dist = length((in.point - in.center) / in.width / 2);
    let alpha = 1.0 - smoothstep(0.82, 1.0, dist);
    let color = vec4(in.color.xyz, alpha * in.color.w);
    return FragmentOutput(color);
}
