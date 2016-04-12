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
        let p: Vec2<f64> = Vec2::new(320f64, 240f64);
        let v: Vec2<f64> = Vec2::new(0f64, 0f64);
        let a: Vec2<f64> = Vec2::new(0f64, 0f64);
        Actor {
            position: p,
            velocity: v,
            acceleration: a,
            genome: neat::Genome::random(6, 2)
        }
    }

    pub fn push(&mut self, force: Vec2<f64>) {
        self.acceleration = self.acceleration + force;
    }

    pub fn update(&mut self) {
        let mut inputs = Vec::new();
        inputs.push(self.position.x);
        inputs.push(self.position.y);
        inputs.push(self.velocity.x);
        inputs.push(self.velocity.y);
        inputs.push(self.acceleration.x);
        inputs.push(self.acceleration.y);

        let outputs = self.genome.network.evaluate(inputs);
        let jerk = Vec2::new(outputs[0], outputs[1]);

        self.acceleration = self.acceleration + jerk * SPT;
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
