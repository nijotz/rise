pub mod neat;

#[macro_use]
extern crate log;
extern crate nalgebra as na;
extern crate rand;
extern crate time;
use na::{Vec2};

pub const TICKS: f64 = 25f64;
// MilliSeconds Per Tick
pub const SPT: f64 = 1f64 / TICKS;

pub struct Actor {
    pub position: Vec2<f64>,
    pub velocity: Vec2<f64>,
    pub acceleration: Vec2<f64>,
    genome: neat::Genome
}

impl Actor {
    pub fn new() -> Actor {
        let p: Vec2<f64> = Vec2::new(0f64, 0f64);
        let v: Vec2<f64> = Vec2::new(0f64, 0f64);
        let a: Vec2<f64> = Vec2::new(1f64, 1f64);
        Actor {
            position: p,
            velocity: v,
            acceleration: a,
            genome: neat::Genome::random(3, 2)
        }
    }

    pub fn push(&mut self, force: Vec2<f64>) {
        self.acceleration = self.acceleration + force;
    }

    pub fn update(&mut self) {
        self.velocity = self.velocity + self.acceleration * SPT;
        self.position = self.position + self.velocity * SPT;
    }
}

pub struct World {
    pub actors: Vec<Actor>,
    subspace: Vec<Vec<Actor>>
}

impl World {
    pub fn new() -> World {
        World {
            actors: Vec::new(),
            subspace: (0..100).map(|_| Vec::new()).collect()
        }
    }

    pub fn update(&mut self) {
        for actor in self.actors.iter_mut() {
            actor.update();
        }
    }
}
