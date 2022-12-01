import {defs, tiny} from './examples/common.js';
import { Shape_From_File } from './examples/obj-file-demo.js';
import { Shadow_Textured_Phong_Shader } from './examples/shadow-demo-shader.js';

const {
    Vector, Vector3, vec, vec3, vec4, color, hex_color, Shader, Matrix, Mat4, Light, Shape, Material, Scene, Texture
} = tiny;

export class FinalProject extends Scene {
   
 // Random number generator
    getRandomNum(min, max) {
        return Math.random() * (max - min) + min;
    }
    
    constructor() {
        // constructor(): Scenes begin by populating initial values like the Shapes and Materials they'll need.
        super();

        const textured = new defs.Textured_Phong(1);

        // Set start key to false
        this.start = false;
        this.jump = false;
        this.jump_height = 20;
        this.difficult = false;
        this.fall_speed = 0.06;
        this.start_time = Number.POSITIVE_INFINITY;

        // At the beginning of our program, load one of each of these shape definitions onto the GPU.
        this.shapes = {
            bear: new defs.Cube(),
            tube: new defs.Cylindrical_Tube(15, 15, [[0, 1], [0, 1]]),
            ground: new defs.Cube(50, 50, [[0, 2], [0, 1]]),
            sphere: new defs.Subdivision_Sphere(4),
            cone: new defs.Closed_Cone(10,10),
            box: new defs.Square(),
            "flappy": new Shape_From_File("assets/flappy.obj")
        };

        // *** Materials
        this.materials = {
            test: new Material(new defs.Phong_Shader(),
                {ambient: .4, diffusivity: .6, color: hex_color("#ffffff")}),
            test2: new Material(new Gouraud_Shader(),
                {ambient: .4, diffusivity: .6, color: hex_color("#992828")}),
            tube: new Material(new defs.Phong_Shader(),
                {ambient: .7, diffusivity: .6, color: hex_color("76C13A")}),
            ground: new Material(new Gouraud_Shader(), 
                {ambient: .3, diffusivity: .9, color: hex_color("#D2B48C")}),
            eye: new Material(new Gouraud_Shader(), 
                {ambient: .3, diffusivity: .9, color: hex_color("#000000")}),
            beak: new Material(new Gouraud_Shader(), 
                {ambient: .3, diffusivity: .9, color: hex_color("#FFD580")}),
            feather: new Material(textured,
                {ambient: 1, diffusivity: 1, specularity: 0,  texture: new Texture("assets/feather.jpg")}),
            sky: new Material(textured,
                {ambient: .9, diffusivity: .9, texture: new Texture("assets/flappy_background.png")}),
            sky2: new Material(textured,
                {ambient: .9, diffusivity: .9, texture: new Texture("assets/sky.png")}),
            dirt: new Material(textured,
                {ambient: .7, diffusivity: .9, texture: new Texture("assets/dirt.png")}),
            flappy: new Material(textured,
                {ambient: .9, diffusivity: .9, texture: new Texture("assets/flappy_bird-00.png")}),
            score0: new Material(textured, 
                {ambient: 1, texture: new Texture("assets/numbers/0.png"), color: color(0, 0, 0, 1)}),
            score1: new Material(textured, 
                {ambient: 1, texture: new Texture("assets/numbers/1.png"), color: color(0, 0, 0, 1)}),
            score2: new Material(textured, 
                {ambient: 1, texture: new Texture("assets/numbers/2.png"), color: color(0, 0, 0, 1)}),
            score3: new Material(textured, 
                {ambient: 1, texture: new Texture("assets/numbers/3.png"), color: color(0, 0, 0, 1)}),
            score4: new Material(textured, 
                {ambient: 1, texture: new Texture("assets/numbers/4.png"), color: color(0, 0, 0, 1)}),
            score5: new Material(textured, 
                {ambient: 1, texture: new Texture("assets/numbers/5.png"), color: color(0, 0, 0, 1)}),
            score6: new Material(textured, 
                {ambient: 1, texture: new Texture("assets/numbers/6.png"), color: color(0, 0, 0, 1)}),
            score7: new Material(textured, 
                {ambient: 1, texture: new Texture("assets/numbers/7.png"), color: color(0, 0, 0, 1)}),
            score8: new Material(textured, 
                {ambient: 1, texture: new Texture("assets/numbers/8.png"), color: color(0, 0, 0, 1)}),
            score9: new Material(textured, 
                {ambient: 1, texture: new Texture("assets/numbers/9.png"), color: color(0, 0, 0, 1)}),
    
        }

        this.floor = new Material(new Gouraud_Shader(), 
        {ambient: 0.3, diffusivity: .9, color: hex_color("#ffaf40")}),
            
        // Number of pipes
        this.NUM_PIPES = 100;
        this.pipe_heights = Array.from({length: this.NUM_PIPES}, () => this.getRandomNum(0, -12)); // height off base level
        this.pipe_gaps = Array.from({length: this.NUM_PIPES}, () => this.getRandomNum(-24, -26)); // gap size between pipes

        // this.initial_camera_location = Mat4.translation(5,-10,-30);
        this.initial_camera_location = Mat4.look_at(vec3(0, 20, 60), vec3(0, 17, 15), vec3(0, 1 , 0));
    }

    reset() {
        this.start = false;
        this.jump = false;
        this.jump_height = 20;
        this.difficult = false;
        this.start_time = Number.POSITIVE_INFINITY;
    }

    make_control_panel() {
        // Draw the scene's buttons, setup their actions and keyboard shortcuts, and monitor live measurements.
        // Start Key
        this.key_triggered_button("Start", ["Enter"], () => {this.start = !this.start});
        this.new_line();

        // Jump Key
        this.key_triggered_button("Jump", [" "], () => {this.jump_height += 2});
        this.new_line();
       
        // Default POV
        this.key_triggered_button("Default view", ["x"], () => this.attached = () => this.default_pov);
        this.new_line();

        // Change POV to bird
        this.key_triggered_button("Bird view", ["v"], () => this.attached = () => this.bird_pov);
        this.new_line();

        // Add multiple difficulties
        this.key_triggered_button("Toggle Hard Mode", ["h"], () => {this.difficult = !this.difficult});
        this.new_line();
        
        this.key_triggered_button("-", ["j"], () => {this.fall_speed = Math.max(0.03, this.fall_speed - 0.01)});
        this.live_string(box => {
                box.textContent = "Fall Speed: " + Math.round((this.fall_speed * 100)) / 100});
        this.key_triggered_button("+", ["u"], () => {this.fall_speed += 0.01});
        this.new_line();

        // Reset button
        this.key_triggered_button("Reset", ["e"], () => {
            this.reset();
        });
        this.new_line();
    }
    
    display(context, program_state) {
        // display():  Called once per frame of animation.
        // Setup -- This part sets up the scene's overall camera matrix, projection matrix, and lights:
        if (!context.scratchpad.controls) {
            this.children.push(context.scratchpad.controls = new defs.Movement_Controls());
            // Define the global camera and projection matrices, which are stored in program_state.
            program_state.set_camera(this.initial_camera_location);
        }
        let t = program_state.animation_time / 1000, dt = program_state.animation_delta_time / 1000;
        let model_transform = Mat4.identity()
        program_state.projection_transform = Mat4.perspective(
            Math.PI / 4, context.width / context.height, .1, 1000);

        // need more lighting for sky backdrop
        const light_position = vec4(0, -10, 0, 1);
        program_state.lights = [new Light(light_position, color(1, 1, 1, 1), 1000)];

        // Draw Bird Avatar
        if (this.start) {
          this.jump_height -= this.fall_speed;
        }
        let model_transform_bird = model_transform.times(Mat4.translation(0, this.jump_height, 0));
        if (this.jump) {
            model_transform_bird = model_transform_bird.times(Mat4.translation(0, this.jump_height, 0))
        }

        let model_transform_bird_pov = model_transform_bird.times(Mat4.rotation(Math.PI / 2,0,1,0)).times(Mat4.translation(0,0,2).times(Mat4.rotation(Math.PI, 0, 1, 0)).times(Mat4.translation(0,0,-7)));
       
        this.bird_pov = model_transform_bird_pov;
        model_transform_bird = model_transform_bird.times(Mat4.scale(1.25,1.25,1.25))
        this.shapes.flappy.draw(context, program_state, model_transform_bird, this.materials.flappy);
        
        // Drawing the ground
        let model_transform_ceiling = model_transform.times(Mat4.rotation(Math.PI/2, 1, 0, 0)).times(Mat4.translation(0, 10, -45)).times(Mat4.scale(1000, 50, 0.5));
        this.shapes.ground.draw(context, program_state, model_transform_ceiling, this.materials.sky2);

        let start_x_ground = -50
        for (let index = 0; index < 25; index++) {
            let model_transform_ground = model_transform.times(Mat4.rotation(Math.PI/2, 1, 0, 0)).times(Mat4.translation(start_x_ground, 10, -3)).times(Mat4.scale(75, 75, 0.5));
            this.shapes.ground.draw(context, program_state, model_transform_ground, this.materials.dirt);
            start_x_ground = start_x_ground + 50;
        }

        let start_x_background = -70
        for (let index = 0; index < 50; index++) {
            let model_transform_sky = model_transform.times(Mat4.translation(start_x_background, 30, -61)).times(Mat4.scale(35, 35, 0.5));
            this.shapes.ground.draw(context, program_state, model_transform_sky, this.materials.sky);
            start_x_background = start_x_background + 70;
        }

        let start_x_background_2 = -70
        for (let index = 0; index < 50; index++) {
            let model_transform_sky = model_transform.times(Mat4.translation(start_x_background_2, 30, 61)).times(Mat4.scale(35, 35, 0.5));
            this.shapes.ground.draw(context, program_state, model_transform_sky, this.materials.sky);
            start_x_background_2 = start_x_background_2 + 70;
        }        
        
       
        let score = 0;
        const MAX_HEIGHT = 20; // total max height for both pipes
        let difficulty_adjustment = -MAX_HEIGHT/8;
        if (!this.difficult) {
            difficulty_adjustment = -MAX_HEIGHT/4;
        }
        
        for (let index = 2; index < this.NUM_PIPES; index ++) { // start from 2 so bird has time to jump
            let pipe_height = this.pipe_heights[index];
            let pipe_gap = this.pipe_gaps[index];
            
            let model_transform_bottom_tube = model_transform.times(Mat4.rotation(Math.PI/2, 1, 0, 0)).times(Mat4.translation(7 * index, 0, pipe_height, 0)).times(Mat4.scale(1, 1, MAX_HEIGHT));
            let model_transform_top_tube = model_transform.times(Mat4.rotation(Math.PI/2, 1, 0, 0)).times(Mat4.translation(7 * index, 0, pipe_height + pipe_gap + difficulty_adjustment, 0)).times(Mat4.scale(1, 1, MAX_HEIGHT));
            
            if (this.start)
            { // need to change this approach so t starts only when the start button is hit
                this.start_time = Math.min(this.start_time, t)
                let shift = -(t - this.start_time)/0.5;
                model_transform_bottom_tube = model_transform_bottom_tube.times(Mat4.translation(shift, 0, 0, 0))
                model_transform_top_tube = model_transform_top_tube.times(Mat4.translation(shift, 0, 0, 0))
            }
            
            let x_coord = model_transform_bottom_tube[0][3];            
            if (x_coord <= 0) {
                score += 1;
            }
            
            if (x_coord <= 0.01 && x_coord >= -0.01) {
                let bird_y_coord = model_transform_bird[1][3];
                let gap_y_coord = -(pipe_height + (pipe_gap + difficulty_adjustment)/2);
                let gap_size = -pipe_gap - MAX_HEIGHT;
                
                // console.log('gap', gap_y_coord)
                // console.log('bird', bird_y_coord)
                // console.log('space', gap_size)
                
                if (Math.abs(bird_y_coord - gap_y_coord) > gap_size/1.5) { // 1.5 for forgiveness
                    console.log('You lose!');
                    this.reset()
                }
            }
            
            this.shapes.tube.draw(context, program_state, model_transform_bottom_tube, this.materials.tube);
            this.shapes.tube.draw(context, program_state, model_transform_top_tube, this.materials.tube);
        }
        
        // console.log(score);
        let n_digits = 0;
        while (score > 0) {
            let last_digit = score % 10;
            let model_transform_score = model_transform.times(Mat4.translation(n_digits * -2, 35, 5, 0));
            
            switch(last_digit) {
                case 0:
                    this.shapes.box.draw(context, program_state, model_transform_score, this.materials.score0)
                    break;
                case 1:
                    this.shapes.box.draw(context, program_state, model_transform_score, this.materials.score1)
                    break;
                case 2:
                    this.shapes.box.draw(context, program_state, model_transform_score, this.materials.score2)
                    break;
                case 3:
                    this.shapes.box.draw(context, program_state, model_transform_score, this.materials.score3)
                    break;
                case 4:
                    this.shapes.box.draw(context, program_state, model_transform_score, this.materials.score4)
                    break;
                case 5:
                    this.shapes.box.draw(context, program_state, model_transform_score, this.materials.score5)
                    break;
                case 6:
                    this.shapes.box.draw(context, program_state, model_transform_score, this.materials.score6)
                    break;
                case 7:
                    this.shapes.box.draw(context, program_state, model_transform_score, this.materials.score7)
                    break;
                case 8:
                    this.shapes.box.draw(context, program_state, model_transform_score, this.materials.score8)
                    break;
                case 9:
                    this.shapes.box.draw(context, program_state, model_transform_score, this.materials.score9)
                    break;
            }
        n_digits += 1;
        score = Math.floor(score/10);
        }
        
        // Resets to our initial world view (initial camera setting) or changes to bird POV
        this.default_pov = this.initial_camera_location;
        if (this.attached != undefined)
        {
            let new_perspective = this.attached();
            if(new_perspective !== this.default_pov)
            {
                new_perspective = new_perspective.times(Mat4.translation(0,0,5));
                new_perspective = Mat4.inverse(new_perspective);
            }
            new_perspective = new_perspective.map((x, i) => Vector.from(program_state.camera_inverse[i]).mix(x,0.1));
            program_state.set_camera(new_perspective);
        }
    }
}



// Custom Shaders
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

// Delete later if needed
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

