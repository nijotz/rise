const TICKS: f64 = 25f64;
// MilliSeconds Per Tick
const SPT: f64 = 1f64 / TICKS;

extern crate nalgebra as na;
extern crate time;
use na::{Vec2};

struct Actor {
    position: Vec2<f64>,
    velocity: Vec2<f64>,
    acceleration: Vec2<f64>
}

impl Actor {
    fn new() -> Actor {
        let p: Vec2<f64> = Vec2::new(0f64, 0f64);
        let v: Vec2<f64> = Vec2::new(0f64, 0f64);
        let a: Vec2<f64> = Vec2::new(1f64, 1f64);
        Actor {
            position: p,
            velocity: v,
            acceleration: a
        }
    }

    fn push(&mut self, force: Vec2<f64>) {
        self.acceleration = self.acceleration + force;
    }

    fn render(&self, gl: &mut GlGraphics, args: &RenderArgs) {
        use graphics::*;
        let square = rectangle::square(0.0, 0.0, 10.0);
        const RED:   [f32; 4] = [1.0, 0.0, 0.0, 1.0];
        let rotation = 0.0;
        gl.draw(args.viewport(), |c, gl| {
            let transform = c.transform.trans(self.position.x, self.position.y)
                                       .rot_rad(rotation)
                                       .trans(-25.0, -25.0);
            rectangle(RED, square, transform, gl)
        })
    }

    fn update(&mut self, args: &UpdateArgs) {
        self.velocity = self.velocity + self.acceleration * SPT;
        self.position = self.position + self.velocity * SPT;
    }
}

extern crate piston;
extern crate graphics;
extern crate glutin_window;
extern crate opengl_graphics;

use piston::window::WindowSettings;
use piston::event_loop::*;
use piston::input::*;
use glutin_window::GlutinWindow as Window;
use opengl_graphics::{ GlGraphics, OpenGL };

pub struct World {
    gl: GlGraphics,
    actors: Vec<Actor>
}

impl World {
    fn render(&mut self, args: &RenderArgs) {
        use graphics::*;
        const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
        self.gl.draw(args.viewport(), |c, gl| {
            // Clear the screen.
            clear(BLACK, gl);
        });

        for actor in &self.actors {
            actor.render(&mut self.gl, args);
        }
    }

    fn update(&mut self, args: &UpdateArgs) {
        for actor in self.actors.iter_mut() {
            actor.update(args);
        }
    }
}

fn main() {
    // Change this to OpenGL::V2_1 if not working.
    let opengl = OpenGL::V3_2;

    // Create an Glutin window.
    let mut window: Window = WindowSettings::new(
            "RISE",
            [200, 200]
        )
        .opengl(opengl)
        .exit_on_esc(true)
        .build()
        .unwrap();

    let mut world = World {
        gl: GlGraphics::new(opengl),
        actors: Vec::new()
    };

    let mut events = window.events();
    while let Some(e) = events.next(&mut window) {
        if let Some(button) = e.press_args() {
            world.actors.push(Actor::new());
        }

        if let Some(r) = e.render_args() {
            world.render(&r);
        }

        if let Some(u) = e.update_args() {
            world.update(&u);
        }
    }
}
