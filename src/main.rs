const TICKS: i32 = 25;
// MilliSeconds Per Tick
const SPT: f64 = 1f64 / 25f64;

extern crate nalgebra as na;
extern crate time;
use std::time::Duration;
use std::mem;
use std::thread;
use na::{Vec2};

struct Actor {
    position: Vec2<f64>,
    velocity: Vec2<f64>,
    acceleration: Vec2<f64>
}

trait Movable {
    fn new() -> Self;
    fn push(&mut self, force: Vec2<f64>) -> ();
    fn draw(&self) -> ();
    fn update(&mut self) -> ();
}

impl Movable for Actor {
    fn new() -> Actor {
        let p: Vec2<f64> = Vec2::new(0f64, 0f64);
        let v: Vec2<f64> = Vec2::new(0f64, 0f64);
        let a: Vec2<f64> = Vec2::new(0f64, 0f64);
        Actor {
            position: p,
            velocity: v,
            acceleration: a
        }
    }

    fn push(&mut self, force: Vec2<f64>) {
        self.acceleration = self.acceleration + force;
    }

    fn draw(&self) {
        println!("position: {}", self.position);
    }

    fn update(&mut self) {
        self.velocity = self.velocity + self.acceleration * SPT;
        self.position = self.position + self.velocity * SPT;
    }
}

fn timestamp () -> f64 {
    let timespec = time::get_time();
    let seconds: f64 = timespec.sec as f64 + (timespec.nsec as f64 / 1000.0 / 1000.0 / 1000.0 );
    seconds
}

fn update(actor: &mut Actor) {
    actor.update();
    actor.draw();
}

fn main() {
    let mut actor: Actor = Actor::new();
    actor.acceleration = Vec2::new(1.0, 1.0);
    let mut next_tick = timestamp();
    let running = true;
    while running {
        if timestamp() > next_tick {
            update(&mut actor);
            next_tick += SPT;
        } else {
            let millis: u64 = ((next_tick - timestamp()) * 1000f64).floor() as u64;
            thread::sleep(Duration::from_millis(millis));
        }
    }
}
