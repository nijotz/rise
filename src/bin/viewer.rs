extern crate env_logger;
extern crate piston;
extern crate graphics;
extern crate glutin_window;
extern crate opengl_graphics;

use piston::window::WindowSettings;
use piston::event_loop::*;
use piston::input::*;
use glutin_window::GlutinWindow as Window;
use opengl_graphics::{ GlGraphics, OpenGL };

// Vector maths
extern crate nalgebra as na;
extern crate time;
use na::{Vec2};

// Simulator
extern crate rise;
use rise::World;
use rise::Actor;

trait Draw {
    fn render(&self, gl: &mut GlGraphics, args: &RenderArgs) -> ();
}

impl Draw for Actor {
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
}

impl Draw for World {
    fn render(&self, mut gl: &mut GlGraphics, args: &RenderArgs) {
        use graphics::*;
        const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
        gl.draw(args.viewport(), |_c, gl| {
            // Clear the screen.
            clear(BLACK, gl);
        });

        for actor in &self.actors {
            actor.render(&mut gl, args);
        }
    }
}

fn main() {
    let _ = env_logger::init();

    // Change this to OpenGL::V2_1 if not working.
    let opengl = OpenGL::V3_2;

    // Create a Glutin window.
    let mut window: Window = WindowSettings::new(
            "RISE",
            [640, 480]
        )
        .opengl(opengl)
        .exit_on_esc(true)
        .build()
        .unwrap();
    let mut gl: GlGraphics = GlGraphics::new(opengl);

    let mut world = World::new();

    let mut events = window.events();
    while let Some(e) = events.next(&mut window) {
        if let Some(_button) = e.press_args() {
            let mut actor = Actor::new();
            actor.push(Vec2::new(1.0, 1.0));
            world.actors.push(actor);
        }

        if let Some(r) = e.render_args() {
            world.render(&mut gl, &r);
        }

        if let Some(_u) = e.update_args() {
            world.update();
        }
    }
}
