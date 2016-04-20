pub mod neat;

use neat::genetics::Genome;
use neat::Creator;

#[macro_use]
extern crate log;
extern crate nalgebra as na;
extern crate rand;
extern crate time;
use na::{Vec2, Pnt2, FloatPnt};

// Updates per second
pub const TICKS: u64 = 25u64;
// Seconds Per Tick
pub const SPT: f64 = 1f64 / TICKS as f64;
// Seconds per generation
pub const SPG: f64 = 10f64;
// Ticks per generation
pub const TPG: u64 = (SPG * TICKS as f64) as u64;

pub struct Actor {
    pub position: Pnt2<f64>,
    pub velocity: Vec2<f64>,
    pub acceleration: Vec2<f64>,
    genome: Genome
}

impl Actor {
    pub fn new(genome: Genome) -> Actor {
        let p: Pnt2<f64> = Pnt2::new(320f64, 240f64);
        let v: Vec2<f64> = Vec2::new(0f64, 0f64);
        let a: Vec2<f64> = Vec2::new(0f64, 0f64);
        Actor {
            position: p,
            velocity: v,
            acceleration: a,
            genome: genome
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
        inputs.push(1f64);

        let outputs = self.genome.network.evaluate(inputs);
        let jerk = Vec2::new(outputs[0], outputs[1]);

        self.acceleration = self.acceleration + jerk * SPT;
        self.velocity = self.velocity + self.acceleration * SPT;
        self.position = self.position + self.velocity * SPT;
    }
}

pub struct World {
    pub actors: Vec<Actor>,
    pub creator: Creator,
    pub generation_tick: u64
}

impl World {
    pub fn new() -> World {
        let mut actors = Vec::with_capacity(100);
        for _ in 0..100 {
            let genome = Genome::random(7, 2);
            let actor = Actor::new(genome);
            actors.push(actor);
        }

        World {
            actors: actors,
            creator: Creator::new(),
            generation_tick: TPG
        }
    }

    pub fn update(&mut self) {
        self.generation_tick -= 1;

        if self.generation_tick <= 0 {
            self.generation_tick = TPG;

            // Get next generation from current actor genomes
            let mut next_gen = self.creator.next_generation(
                self.actors.iter().map(|actor| &actor.genome).collect()
            );

            // Kill your parents
            self.actors.clear();

            // Create actors with new genomes
            for genome in next_gen {
                let actor = Actor::new(genome);
                self.actors.push(actor);
            }
        }

        for actor in self.actors.iter_mut() {
            actor.update();
        }
    }

    //fn actor_genomes(&self) -> Vec<&Genome> {
    //    let mut genomes = Vec::with_capacity(self.actors.len());
    //    for actor in self.actors.iter_mut() { genomes.push(&actor.genome); }
    //    return genomes;
    //}
}

fn fitness(actor: &Actor) -> f64 {
    -actor.position.dist(&Pnt2::new(0f64, 0f64))
}
