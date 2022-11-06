import {defs, tiny} from './common.js';

const {
    Vector, Vector3, vec, vec3, vec4, color, hex_color, Shader, Matrix, Mat4, Light, Shape, Material, Scene,
} = tiny;

export class FinalProject extends Scene {
    constructor() {
        // constructor(): Scenes begin by populating initial values like the Shapes and Materials they'll need.
        super();

        // At the beginning of our program, load one of each of these shape definitions onto the GPU.
        this.shapes = {
            torus: new defs.Torus(15, 15),
            torus2: new defs.Torus(3, 15),
            f1: new (defs.Subdivision_Sphere.prototype.make_flat_shaded_version())(1),
            f2: new (defs.Subdivision_Sphere.prototype.make_flat_shaded_version())(2),
            s3: new defs.Subdivision_Sphere(3),
            s4: new defs.Subdivision_Sphere(4),
            circle: new defs.Regular_2D_Polygon(1, 15),
            // TODO:  Fill in as many additional shape instances as needed in this key/value table.
            //        (Requirement 1)
        };

        // *** Materials

        this.materials = {
            test: new Material(new defs.Phong_Shader(),
                {ambient: .4, diffusivity: .6, color: hex_color("#ffffff")}),
            test2: new Material(new Gouraud_Shader(),
                {ambient: .4, diffusivity: .6, color: hex_color("#992828")}),
            ring: new Material(new Ring_Shader(), {ambient: 1, diffusivity: 0, specularity: 0, color: hex_color("#ffffff")}),
            sun: new Material(new defs.Phong_Shader(),
                {ambient: 1, diffusivity: 0, specularity: 0, color: hex_color("#ffffff")}),
            moon: new Material(new defs.Phong_Shader(),
                {ambient: 1, diffusivity: 1, specularity: 0, color: hex_color("#C7E4EE")}),
            p1: new Material(new defs.Phong_Shader(),
                {ambient: 0, diffusivity: 1, specularity: 0, color: hex_color("#808080")}),
            p2: new Material(new defs.Phong_Shader(),
                {ambient: 0, diffusivity: 0.25, specularity: 1, color: hex_color("#80FFFF")}),
            p2_Gourad: new Material(new Gouraud_Shader(),
                {ambient: 0, diffusivity: 0.25, specularity: 1, color: hex_color("#80FFFF")}),
            p3: new Material(new defs.Phong_Shader(),
                {ambient: 0, diffusivity: 1, specularity: 1, color: hex_color("#B08040")}),
            p4: new Material(new defs.Phong_Shader(),
                {ambient: 0, diffusivity: 0, specularity: 1, color: hex_color("#C7E4EE")}),
            
            // TODO:  Fill in as many additional material objects as needed in this key/value table.
            //        (Requirement 4)
        }

        this.initial_camera_location = Mat4.look_at(vec3(0, 10, 20), vec3(0, 0, 0), vec3(0, 1, 0));
    }

    make_control_panel() {
        // Draw the scene's buttons, setup their actions and keyboard shortcuts, and monitor live measurements.
        this.key_triggered_button("View solar system", ["Control", "0"], () => this.attached = () => this.planet_0);
        this.new_line();
        this.key_triggered_button("Attach to planet 1", ["Control", "1"], () => this.attached = () => this.planet_1);
        this.key_triggered_button("Attach to planet 2", ["Control", "2"], () => this.attached = () => this.planet_2);
        this.new_line();
        this.key_triggered_button("Attach to planet 3", ["Control", "3"], () => this.attached = () => this.planet_3);
        this.key_triggered_button("Attach to planet 4", ["Control", "4"], () => this.attached = () => this.planet_4);
        this.new_line();
        this.key_triggered_button("Attach to moon", ["Control", "m"], () => this.attached = () => this.moon);
    }

    display(context, program_state) {
        // display():  Called once per frame of animation.
        // Setup -- This part sets up the scene's overall camera matrix, projection matrix, and lights:
        if (!context.scratchpad.controls) {
            this.children.push(context.scratchpad.controls = new defs.Movement_Controls());
            // Define the global camera and projection matrices, which are stored in program_state.
            program_state.set_camera(this.initial_camera_location);
        }
        let model_transform = Mat4.identity()
        program_state.projection_transform = Mat4.perspective(
            Math.PI / 4, context.width / context.height, .1, 1000);
        
        // Resets to our initial solar system view (initial camera setting)
        this.planet_0 = this.initial_camera_location;
        if (this.attached != undefined)
        {
            // Requirement 6 Implementation
            let new_perspective = this.attached();
            if(new_perspective !== this.planet_0)
            {
                new_perspective = new_perspective.times(Mat4.translation(0,0,5));
                new_perspective = Mat4.inverse(new_perspective);
            }
            new_perspective = new_perspective.map((x, i) => Vector.from(program_state.camera_inverse[i]).mix(x,0.1));
            program_state.set_camera(new_perspective);
        }

        // TODO: Create Planets (Requirement 1)
        // this.shapes.[XXX].draw([XXX]) // <--example
        const t = program_state.animation_time / 2000, dt = program_state.animation_delta_time / 1000;
        let sun_radius = 2 + Math.sin((Math.PI * 2/5)*t - (Math.PI/2));
        // TODO: Lighting (Requirement 2)
        const light_position = vec4(0, 5, 5, 1);
        // The parameters of the Light are: position, color, size
        program_state.lights = [new Light(light_position, color(1, -0.5 + sun_radius / 2, -0.5 + sun_radius / 2, 1 ), 10**sun_radius)];

        // TODO:  Fill in matrix operations and drawing code to draw the solar system scene (Requirements 3 and 4)
        
        model_transform = Mat4.identity().times(Mat4.scale(sun_radius, sun_radius, sun_radius));
        const swampy = hex_color("#80FFFF")
        const gray = hex_color("#808080")
        const muddy = hex_color("#B08040")
        const soft_blue = hex_color("#C7E4EE")
        let sun_color = color(1, -0.5 + sun_radius / 2, -0.5 + sun_radius / 2, 1)

        // Draw Sun
        this.shapes.s4.draw(context, program_state, model_transform, this.materials.sun.override({color: sun_color}));
        
        // Draw First Planet
        model_transform = model_transform.times(Mat4.scale(1 / sun_radius, 1 / sun_radius, 1 / sun_radius).times(Mat4.rotation(t, 0, 1, 0))).times(Mat4.translation(5, 0, 0));
        this.shapes.f2.draw(context, program_state, model_transform, this.materials.p1.override({color: gray, diffusivity: 1}));
        
        this.planet_1 = model_transform;

        // Draw Second Planet
        model_transform = model_transform.times(Mat4.translation(-5, 0, 0)).times(Mat4.rotation(-t, 0, 1, 0)).times(Mat4.rotation(0.9*t, 0, 1, 0)).times(Mat4.translation(8,0,0));

        // Alternate Shader on Even/Odd Seconds
        if (parseInt(t) % 2 == 1) {
            this.shapes.s3.draw(context, program_state, model_transform, this.materials.p2_Gourad.override({color: swampy, diffusivity: 0.25, specularity: 1}));
        } else {
            this.shapes.s3.draw(context, program_state, model_transform, this.materials.p2.override({color: swampy, diffusivity: 0.25, specularity: 1}));
        }

        this.planet_2 = model_transform;
        
        // Draw Third Planet
        model_transform = model_transform.times(Mat4.translation(-8, 0, 0)).times(Mat4.rotation(-0.9*t, 0, 1, 0)).times(Mat4.rotation(0.5*t, 0, 1, 0)).times(Mat4.translation(11,0,0));
        this.shapes.s4.draw(context, program_state, model_transform, this.materials.p3.override({color: muddy, diffusivity: 1, specularity: 0.5}));

        this.planet_3 = model_transform;

        // Draw Ring
        model_transform = model_transform.times(Mat4.scale(3, 3, 0.2))
        this.shapes.torus.draw(context, program_state, model_transform, this.materials.ring);

        // Draw Fourth Planet
        model_transform = model_transform.times(Mat4.scale(1 / 3, 1 / 3, 5)).times(Mat4.translation(-11, 0, 0)).times(Mat4.rotation(-0.5*t, 0, 1, 0)).times(Mat4.rotation(0.25*t, 0, 1, 0)).times(Mat4.translation(14,0,0));
        this.shapes.s4.draw(context, program_state, model_transform, this.materials.p4.override({color: soft_blue, specularity: 1}));

        this.planet_4 = model_transform;

        // Draw Moon
        model_transform = model_transform.times(Mat4.rotation(2 * t, 0, 1, 0)).times(Mat4.translation(2.5,0,0));
        this.shapes.f1.draw(context, program_state, model_transform, this.materials.moon.override({color: soft_blue}));

        this.moon = model_transform;
        
    }
}

class Gouraud_Shader extends Shader {
    // This is a Shader using Phong_Shader as template
    // TODO: Modify the glsl coder here to create a Gouraud Shader (Planet 2)

    constructor(num_lights = 2) {
        super();
        this.num_lights = num_lights;
    }

    shared_glsl_code() {
        // ********* SHARED CODE, INCLUDED IN BOTH SHADERS *********
        
        return ` 
        precision mediump float;
        const int N_LIGHTS = ` + this.num_lights + `;
        uniform float ambient, diffusivity, specularity, smoothness;
        uniform vec4 light_positions_or_vectors[N_LIGHTS], light_colors[N_LIGHTS];
        uniform float light_attenuation_factors[N_LIGHTS];
        uniform vec4 shape_color;
        uniform vec3 squared_scale, camera_center;
        varying vec3 storing_lights;
        // Specifier "varying" means a variable's final value will be passed from the vertex shader
        // on to the next phase (fragment shader), then interpolated per-fragment, weighted by the
        // pixel fragment's proximity to each of the 3 vertices (barycentric interpolation).
        varying vec3 N, vertex_worldspace;
        // ***** PHONG SHADING HAPPENS HERE: *****                                       
        vec3 phong_model_lights( vec3 N, vec3 vertex_worldspace ){                                        
            // phong_model_lights():  Add up the lights' contributions.
            vec3 E = normalize( camera_center - vertex_worldspace );
            vec3 result = vec3( 0.0 );
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
                result += attenuation * light_contribution;
            }
            return result;
        } `;
    }

    vertex_glsl_code() {
        // ********* VERTEX SHADER *********
        return this.shared_glsl_code() + `
            attribute vec3 position, normal;                            
            // Position is expressed in object coordinates.
            
            uniform mat4 model_transform;
            uniform mat4 projection_camera_model_transform;
    
            void main(){

                // The vertex's final resting place (in NDCS):
                gl_Position = projection_camera_model_transform * vec4( position, 1.0 );
                // The final normal vector in screen space.
                N = normalize( mat3( model_transform ) * normal / squared_scale);
                vertex_worldspace = ( model_transform * vec4( position, 1.0 ) ).xyz;
                storing_lights = phong_model_lights( normalize( N ), vertex_worldspace );
            } `;
    }

    fragment_glsl_code() {
        // ********* FRAGMENT SHADER *********
        // A fragment is a pixel that's overlapped by the current triangle.
        // Fragments affect the final image or get discarded due to depth.
        return this.shared_glsl_code() + `
            void main(){                                                           
                // Compute an initial (ambient) color:
                gl_FragColor = vec4( shape_color.xyz * ambient, shape_color.w );
                // Compute the final color with contributions from lights:
                gl_FragColor.xyz += storing_lights;
            } `;
    }

    send_material(gl, gpu, material) {
        // send_material(): Send the desired shape-wide material qualities to the
        // graphics card, where they will tweak the Phong lighting formula.
        gl.uniform4fv(gpu.shape_color, material.color);
        gl.uniform1f(gpu.ambient, material.ambient);
        gl.uniform1f(gpu.diffusivity, material.diffusivity);
        gl.uniform1f(gpu.specularity, material.specularity);
        gl.uniform1f(gpu.smoothness, material.smoothness);
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
        const PCM = gpu_state.projection_transform.times(gpu_state.camera_inverse).times(model_transform);
        gl.uniformMatrix4fv(gpu.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        gl.uniformMatrix4fv(gpu.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));

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
        // update_GPU(): Define how to synchronize our JavaScript's variables to the GPU's.  This is where the shader
        // recieves ALL of its inputs.  Every value the GPU wants is divided into two categories:  Values that belong
        // to individual objects being drawn (which we call "Material") and values belonging to the whole scene or
        // program (which we call the "Program_State").  Send both a material and a program state to the shaders
        // within this function, one data field at a time, to fully initialize the shader for a draw.

        // Fill in any missing fields in the Material object with custom defaults for this shader:
        const defaults = {color: color(0, 0, 0, 1), ambient: 0, diffusivity: 1, specularity: 1, smoothness: 40};
        material = Object.assign({}, defaults, material);

        this.send_material(context, gpu_addresses, material);
        this.send_gpu_state(context, gpu_addresses, gpu_state, model_transform);
    }
}

class Ring_Shader extends Shader {
    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        // update_GPU():  Defining how to synchronize our JavaScript's variables to the GPU's:
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform],
            PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false,
            Matrix.flatten_2D_to_1D(PCM.transposed()));
    }

    shared_glsl_code() {
        // ********* SHARED CODE, INCLUDED IN BOTH SHADERS *********
        return `
        precision mediump float;
        varying vec4 position_WCS;
        varying vec4 center;
        
        `;
    }

    vertex_glsl_code() {
        // ********* VERTEX SHADER *********
        // TODO:  Complete the main function of the vertex shader (Extra Credit Part II).
        return this.shared_glsl_code() + `
        attribute vec3 position;
        uniform mat4 model_transform;
        uniform mat4 projection_camera_model_transform;
        varying vec4 vec4position;
        
        void main(){
            
            gl_Position = projection_camera_model_transform * vec4( position, 1.0 );
            position_WCS = model_transform * vec4( position, 1.0 );
            center = model_transform * vec4(0.0, 0.0, 0.0, 1.0);
        }`;
    }

    fragment_glsl_code() {
        // ********* FRAGMENT SHADER *********
        // TODO:  Complete the main function of the fragment shader (Extra Credit Part II).
        return this.shared_glsl_code() + `
        void main(){
            float m_distance = length(position_WCS.xyz - center.xyz);
            float factor = 0.5 + 0.5 * sin(m_distance * 30.0);
        
            gl_FragColor = factor * vec4(0.690, 0.502, 0.251, 1.0/factor);
        }`;
    }
}

