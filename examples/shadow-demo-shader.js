import {defs, tiny} from './common.js';
// Pull these names into this module's scope for convenience:
const {vec3, vec4, vec, color, Matrix, Mat4, Light, Shape, Material, Shader, Texture, Scene} = tiny;
const {Cube, Axis_Arrows, Textured_Phong, Phong_Shader, Basic_Shader, Subdivision_Sphere} = defs

// The size of the light texture buffer
export const LIGHT_DEPTH_TEX_SIZE = 2048;

export class Color_Phong_Shader extends defs.Phong_Shader {
        vertex_glsl_code() {
            // ********* VERTEX SHADER *********
            return this.shared_glsl_code() + `
                varying vec2 f_tex_coord;
                attribute vec3 position, normal;                            
                // Position is expressed in object coordinates.
                attribute vec2 texture_coord;
                
                uniform mat4 model_transform;
                uniform mat4 projection_camera_model_transform;
        
                void main(){                                                                   
                    // The vertex's final resting place (in NDCS):
                    gl_Position = projection_camera_model_transform * vec4( position, 1.0 );
                    // The final normal vector in screen space.
                    N = normalize( mat3( model_transform ) * normal / squared_scale);
                    vertex_worldspace = ( model_transform * vec4( position, 1.0 ) ).xyz;
                    // Turn the per-vertex texture coordinate into an interpolated variable.
                    f_tex_coord = texture_coord;
                  } `;
        }

        fragment_glsl_code() {
            // ********* FRAGMENT SHADER *********
            // A fragment is a pixel that's overlapped by the current triangle.
            // Fragments affect the final image or get discarded due to depth.
            return this.shared_glsl_code() + `
                uniform sampler2D texture;
                uniform sampler2D light_depth_texture;
                uniform mat4 light_view_mat;
                uniform mat4 light_proj_mat;
                
                void main(){
                    gl_FragColor = vec4( (shape_color.xyz ) * ambient, shape_color.w ); 
                                                                             // Compute the final color with contributions from lights:
                    gl_FragColor.xyz += phong_model_lights( normalize( N ), vertex_worldspace );
                  } `;
        }

        send_gpu_state(gl, gpu, gpu_state, model_transform) {
            // send_gpu_state():  Send the state of our whole drawing context to the GPU.
            const O = vec4(0, 0, 0, 1), camera_center = gpu_state.camera_transform.times(O).to3();
            gl.uniform3fv(gpu.camera_center, camera_center);
            // Use the squared scale trick from "Eric's blog" instead of inverse transpose matrix:
            const squared_scale = model_transform.reduce(
                (acc, r) => {
                    return acc.plus(vec4(...r).times_pairwise(r))
                }, vec4(0, 0, 0, 0)).to3();
            gl.uniform3fv(gpu.squared_scale, squared_scale);
            // Send the current matrices to the shader.  Go ahead and pre-compute
            // the products we'll need of the of the three special matrices and just
            // cache and send those.  They will be the same throughout this draw
            // call, and thus across each instance of the vertex shader.
            // Transpose them since the GPU expects matrices as column-major arrays.
            const PCM = gpu_state.projection_transform.times(gpu_state.view_mat).times(model_transform);
            gl.uniformMatrix4fv(gpu.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
            gl.uniformMatrix4fv(gpu.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));
            // shadow related
            gl.uniformMatrix4fv(gpu.light_view_mat, false, Matrix.flatten_2D_to_1D(gpu_state.light_view_mat.transposed()));
            gl.uniformMatrix4fv(gpu.light_proj_mat, false, Matrix.flatten_2D_to_1D(gpu_state.light_proj_mat.transposed()));

            // Omitting lights will show only the material color, scaled by the ambient term:
            if (!gpu_state.lights.length)
                return;

            const light_positions_flattened = [], light_colors_flattened = [];
            for (let i = 0; i < 4 * gpu_state.lights.length; i++) {
                light_positions_flattened.push(gpu_state.lights[Math.floor(i / 4)].position[i % 4]);
                light_colors_flattened.push(gpu_state.lights[Math.floor(i / 4)].color[i % 4]);
            }
            gl.uniform4fv(gpu.light_positions_or_vectors, light_positions_flattened);
            gl.uniform4fv(gpu.light_colors, light_colors_flattened);
            gl.uniform1fv(gpu.light_attenuation_factors, gpu_state.lights.map(l => l.attenuation));
        }

        update_GPU(context, gpu_addresses, gpu_state, model_transform, material) {
            // update_GPU(): Add a little more to the base class's version of this method.
            super.update_GPU(context, gpu_addresses, gpu_state, model_transform, material);
            // Updated for assignment 4
            context.uniform1f(gpu_addresses.animation_time, gpu_state.animation_time / 1000);
        }
    }

export class Shadow_Textured_Phong_Shader extends defs.Phong_Shader {
        shared_glsl_code() {
            // ********* SHARED CODE, INCLUDED IN BOTH SHADERS *********
            return ` precision mediump float;
                const int N_LIGHTS = ` + this.num_lights + `;
                uniform float ambient, diffusivity, specularity, smoothness;
                uniform vec4 light_positions_or_vectors[N_LIGHTS], light_colors[N_LIGHTS];
                uniform float light_attenuation_factors[N_LIGHTS];
                uniform vec4 shape_color;
                uniform vec3 squared_scale, camera_center;
        
                // Specifier "varying" means a variable's final value will be passed from the vertex shader
                // on to the next phase (fragment shader), then interpolated per-fragment, weighted by the
                // pixel fragment's proximity to each of the 3 vertices (barycentric interpolation).
                varying vec3 N, vertex_worldspace;
                // ***** PHONG SHADING HAPPENS HERE: *****                                       
                vec3 phong_model_lights( vec3 N, vec3 vertex_worldspace, 
                        out vec3 light_diffuse_contribution, out vec3 light_specular_contribution ){                                        
                    // phong_model_lights():  Add up the lights' contributions.
                    vec3 E = normalize( camera_center - vertex_worldspace );
                    vec3 result = vec3( 0.0 );
                    light_diffuse_contribution = vec3( 0.0 );
                    light_specular_contribution = vec3( 0.0 );
                    for(int i = 0; i < N_LIGHTS; i++){
                        // Lights store homogeneous coords - either a position or vector.  If w is 0, the 
                        // light will appear directional (uniform direction from all points), and we 
                        // simply obtain a vector towards the light by directly using the stored value.
                        // Otherwise if w is 1 it will appear as a point light -- compute the vector to 
                        // the point light's location from the current surface point.  In either case, 
                        // fade (attenuate) the light as the vector needed to reach it gets longer.  
                        vec3 surface_to_light_vector = light_positions_or_vectors[i].xyz - 
                                                       light_positions_or_vectors[i].w * vertex_worldspace;                                             
                        float distance_to_light = length( surface_to_light_vector );
        
                        vec3 L = normalize( surface_to_light_vector );
                        vec3 H = normalize( L + E );
                        // Compute the diffuse and specular components from the Phong
                        // Reflection Model, using Blinn's "halfway vector" method:
                        float diffuse  =      max( dot( N, L ), 0.0 );
                        float specular = pow( max( dot( N, H ), 0.0 ), smoothness );
                        float attenuation = 1.0 / (1.0 + light_attenuation_factors[i] * distance_to_light * distance_to_light );
                        
                        vec3 light_contribution = shape_color.xyz * light_colors[i].xyz * diffusivity * diffuse
                                                                  + light_colors[i].xyz * specularity * specular;
                        light_diffuse_contribution += attenuation * shape_color.xyz * light_colors[i].xyz * diffusivity * diffuse;
                        light_specular_contribution += attenuation * shape_color.xyz * specularity * specular;
                        result += attenuation * light_contribution;
                      }
                    return result;
                  } `;
        }
        vertex_glsl_code() {
            // ********* VERTEX SHADER *********
            return this.shared_glsl_code() + `
                varying vec2 f_tex_coord;
                attribute vec3 position, normal;                            
                // Position is expressed in object coordinates.
                attribute vec2 texture_coord;
                
                uniform mat4 model_transform;
                uniform mat4 projection_camera_model_transform;
        
                void main(){                                                                   
                    // The vertex's final resting place (in NDCS):
                    gl_Position = projection_camera_model_transform * vec4( position, 1.0 );
                    // The final normal vector in screen space.
                    N = normalize( mat3( model_transform ) * normal / squared_scale);
                    vertex_worldspace = ( model_transform * vec4( position, 1.0 ) ).xyz;
                    // Turn the per-vertex texture coordinate into an interpolated variable.
                    f_tex_coord = texture_coord;
                  } `;
        }

        fragment_glsl_code() {
            // ********* FRAGMENT SHADER *********
            // A fragment is a pixel that's overlapped by the current triangle.
            // Fragments affect the final image or get discarded due to depth.
            return this.shared_glsl_code() + `
                varying vec2 f_tex_coord;
                uniform sampler2D texture;
                uniform sampler2D light_depth_texture;
                uniform mat4 light_view_mat;
                uniform mat4 light_proj_mat;
                uniform float animation_time;
                uniform float light_depth_bias;
                uniform bool use_texture;
                uniform bool draw_shadow;
                uniform float light_texture_size;
                
                float PCF_shadow(vec2 center, float projected_depth) {
                    float shadow = 0.0;
                    float texel_size = 1.0 / light_texture_size;
                    for(int x = -1; x <= 1; ++x)
                    {
                        for(int y = -1; y <= 1; ++y)
                        {
                            float light_depth_value = texture2D(light_depth_texture, center + vec2(x, y) * texel_size).r; 
                            shadow += projected_depth >= light_depth_value + light_depth_bias ? 1.0 : 0.0;        
                        }    
                    }
                    shadow /= 9.0;
                    return shadow;
                }
                
                void main(){
                    // Sample the texture image in the correct place:
                    vec4 tex_color = texture2D( texture, f_tex_coord );
                    if (!use_texture)
                        tex_color = vec4(0, 0, 0, 1);
                    if( tex_color.w < .01 ) discard;
                    
                    // Compute an initial (ambient) color:
                    gl_FragColor = vec4( ( tex_color.xyz + shape_color.xyz ) * ambient, shape_color.w * tex_color.w ); 
                    
                    // Compute the final color with contributions from lights:
                    vec3 diffuse, specular;
                    vec3 other_than_ambient = phong_model_lights( normalize( N ), vertex_worldspace, diffuse, specular );
                    
                    // Deal with shadow:
                    if (draw_shadow) {
                        vec4 light_tex_coord = (light_proj_mat * light_view_mat * vec4(vertex_worldspace, 1.0));
                        // convert NDCS from light's POV to light depth texture coordinates
                        light_tex_coord.xyz /= light_tex_coord.w; 
                        light_tex_coord.xyz *= 0.5;
                        light_tex_coord.xyz += 0.5;
                        float light_depth_value = texture2D( light_depth_texture, light_tex_coord.xy ).r;
                        float projected_depth = light_tex_coord.z;
                        
                        bool inRange =
                            light_tex_coord.x >= 0.0 &&
                            light_tex_coord.x <= 1.0 &&
                            light_tex_coord.y >= 0.0 &&
                            light_tex_coord.y <= 1.0;
                              
                        float shadowness = PCF_shadow(light_tex_coord.xy, projected_depth);
                        
                        if (inRange && shadowness > 0.3) {
                            diffuse *= 0.2 + 0.8 * (1.0 - shadowness);
                            specular *= 1.0 - shadowness;
                        }
                    }
                    
                    gl_FragColor.xyz += diffuse + specular;
                } `;
        }

        send_gpu_state(gl, gpu, gpu_state, model_transform) {
            // send_gpu_state():  Send the state of our whole drawing context to the GPU.
            const O = vec4(0, 0, 0, 1), camera_center = gpu_state.camera_transform.times(O).to3();
            gl.uniform3fv(gpu.camera_center, camera_center);
            // Use the squared scale trick from "Eric's blog" instead of inverse transpose matrix:
            const squared_scale = model_transform.reduce(
                (acc, r) => {
                    return acc.plus(vec4(...r).times_pairwise(r))
                }, vec4(0, 0, 0, 0)).to3();
            gl.uniform3fv(gpu.squared_scale, squared_scale);
            // Send the current matrices to the shader.  Go ahead and pre-compute
            // the products we'll need of the of the three special matrices and just
            // cache and send those.  They will be the same throughout this draw
            // call, and thus across each instance of the vertex shader.
            // Transpose them since the GPU expects matrices as column-major arrays.
            const PCM = gpu_state.projection_transform.times(gpu_state.view_mat).times(model_transform);
            gl.uniformMatrix4fv(gpu.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
            gl.uniformMatrix4fv(gpu.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));
            // shadow related
            gl.uniformMatrix4fv(gpu.light_view_mat, false, Matrix.flatten_2D_to_1D(gpu_state.light_view_mat.transposed()));
            gl.uniformMatrix4fv(gpu.light_proj_mat, false, Matrix.flatten_2D_to_1D(gpu_state.light_proj_mat.transposed()));

            // Omitting lights will show only the material color, scaled by the ambient term:
            if (!gpu_state.lights.length)
                return;

            const light_positions_flattened = [], light_colors_flattened = [];
            for (let i = 0; i < 4 * gpu_state.lights.length; i++) {
                light_positions_flattened.push(gpu_state.lights[Math.floor(i / 4)].position[i % 4]);
                light_colors_flattened.push(gpu_state.lights[Math.floor(i / 4)].color[i % 4]);
            }
            gl.uniform4fv(gpu.light_positions_or_vectors, light_positions_flattened);
            gl.uniform4fv(gpu.light_colors, light_colors_flattened);
            gl.uniform1fv(gpu.light_attenuation_factors, gpu_state.lights.map(l => l.attenuation));
        }

        update_GPU(context, gpu_addresses, gpu_state, model_transform, material) {
            // update_GPU(): Add a little more to the base class's version of this method.
            super.update_GPU(context, gpu_addresses, gpu_state, model_transform, material);
            // Updated for assignment 4
            context.uniform1f(gpu_addresses.animation_time, gpu_state.animation_time / 1000);
            if (material.color_texture && material.color_texture.ready) {
                // Select texture unit 0 for the fragment shader Sampler2D uniform called "texture":
                context.uniform1i(gpu_addresses.color_texture, 0); // 0 for color texture
                // For this draw, use the texture image from correct the GPU buffer:
                context.activeTexture(context["TEXTURE" + 0]);
                material.color_texture.activate(context);
                context.uniform1i(gpu_addresses.use_texture, 1);
            }
            else {
                context.uniform1i(gpu_addresses.use_texture, 0);
            }
            if (gpu_state.draw_shadow) {
                context.uniform1i(gpu_addresses.draw_shadow, 1);
                context.uniform1f(gpu_addresses.light_depth_bias, 0.003);
                context.uniform1f(gpu_addresses.light_texture_size, LIGHT_DEPTH_TEX_SIZE);
                context.uniform1i(gpu_addresses.light_depth_texture, 1); // 1 for light-view depth texture}
                if (material.light_depth_texture && material.light_depth_texture.ready) {
                    context.activeTexture(context["TEXTURE" + 1]);
                    material.light_depth_texture.activate(context, 1);
                }
            }
            else {
                context.uniform1i(gpu_addresses.draw_shadow, 0);
            }
        }
    }

export class Depth_Texture_Shader_2D extends defs.Phong_Shader {
        // **Textured_Phong** is a Phong Shader extended to addditionally decal a
        // texture image over the drawn shape, lined up according to the texture
        // coordinates that are stored at each shape vertex.

        vertex_glsl_code() {
            // ********* VERTEX SHADER *********
            return this.shared_glsl_code() + `
                varying vec2 f_tex_coord;
                attribute vec3 position, normal;                            
                // Position is expressed in object coordinates.
                attribute vec2 texture_coord;
                
                uniform mat4 model_transform;
                uniform mat4 projection_camera_model_transform;
        
                void main(){                                                                   
                    // The vertex's final resting place (in NDCS):
                    gl_Position = model_transform * vec4( position.xy, -1, 1.0 ); // <== only Model, no View
                    // The final normal vector in screen space.
                    N = normalize( mat3( model_transform ) * normal / squared_scale);
                    vertex_worldspace = ( model_transform * vec4( position, 1.0 ) ).xyz;
                    // Turn the per-vertex texture coordinate into an interpolated variable.
                    f_tex_coord = texture_coord;
                  } `;
        }

        fragment_glsl_code() {
            // ********* FRAGMENT SHADER *********
            // A fragment is a pixel that's overlapped by the current triangle.
            // Fragments affect the final image or get discarded due to depth.
            return this.shared_glsl_code() + `
                varying vec2 f_tex_coord;
                uniform sampler2D texture;
                uniform float animation_time;
                
                void main(){
                    // Sample the texture image in the correct place:
                    vec4 tex_color = texture2D( texture, f_tex_coord );
                    tex_color.y = tex_color.z = tex_color.x;
                    if( tex_color.w < .01 ) discard;
                                                                             // Compute an initial (ambient) color:
                    gl_FragColor = vec4( ( tex_color.xyz + shape_color.xyz ) * ambient, shape_color.w * tex_color.w ); 
                                                                             // Compute the final color with contributions from lights:
                    gl_FragColor.xyz += phong_model_lights( normalize( N ), vertex_worldspace );
                  } `;
        }

        update_GPU(context, gpu_addresses, gpu_state, model_transform, material) {
            // update_GPU(): Add a little more to the base class's version of this method.
            super.update_GPU(context, gpu_addresses, gpu_state, model_transform, material);
            // Updated for assignment 4
            context.uniform1f(gpu_addresses.animation_time, gpu_state.animation_time / 1000);
            context.uniform1i(gpu_addresses.texture, 2); // 2 for light-view depth texture
            context.activeTexture(context["TEXTURE" + 2]);
            context.bindTexture(context.TEXTURE_2D, material.texture);
        }
    }

export class Buffered_Texture extends tiny.Graphics_Card_Object {
        // **Texture** wraps a pointer to a new texture image where
        // it is stored in GPU memory, along with a new HTML image object.
        // This class initially copies the image to the GPU buffers,
        // optionally generating mip maps of it and storing them there too.
        constructor(texture_buffer_pointer) {
            super();
            Object.assign(this, {texture_buffer_pointer});
            this.ready = true;
            this.texture_buffer_pointer = texture_buffer_pointer;
        }

        copy_onto_graphics_card(context, need_initial_settings = true) {
            // copy_onto_graphics_card():  Called automatically as needed to load the
            // texture image onto one of your GPU contexts for its first time.

            // Define what this object should store in each new WebGL Context:
            const initial_gpu_representation = {texture_buffer_pointer: undefined};
            // Our object might need to register to multiple GPU contexts in the case of
            // multiple drawing areas.  If this is a new GPU context for this object,
            // copy the object to the GPU.  Otherwise, this object already has been
            // copied over, so get a pointer to the existing instance.
            const gpu_instance = super.copy_onto_graphics_card(context, initial_gpu_representation);

            if (!gpu_instance.texture_buffer_pointer) gpu_instance.texture_buffer_pointer = this.texture_buffer_pointer;

            // const gl = context;
            // gl.bindTexture(gl.TEXTURE_2D, gpu_instance.texture_buffer_pointer);
            //
            // if (need_initial_settings) {
            //     gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
            //     gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            //     // Always use bi-linear sampling when zoomed out.
            //     gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl[this.min_filter]);
            //     // Let the user to set the sampling method
            //     // when zoomed in.
            // }
            //
            // gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.image);
            // if (this.min_filter == "LINEAR_MIPMAP_LINEAR")
            //     gl.generateMipmap(gl.TEXTURE_2D);
            // // If the user picked tri-linear sampling (the default) then generate
            // // the necessary "mips" of the texture and store them on the GPU with it.
            return gpu_instance;
        }

        activate(context, texture_unit = 0) {
            // activate(): Selects this Texture in GPU memory so the next shape draws using it.
            // Optionally select a texture unit in case you're using a shader with many samplers.
            // Terminate draw requests until the image file is actually loaded over the network:
            if (!this.ready)
                return;
            const gpu_instance = super.activate(context);
            context.activeTexture(context["TEXTURE" + texture_unit]);
            context.bindTexture(context.TEXTURE_2D, this.texture_buffer_pointer);
        }
    }