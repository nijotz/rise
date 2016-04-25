pub mod genetics;
pub mod neurology;
pub mod taxonomy;

use neat::genetics::Genome;
use neat::taxonomy::Species;

use rand::{SeedableRng, StdRng};
static RNG_SEED: &'static[usize] = &[294367];

pub fn rng() -> StdRng {
    SeedableRng::from_seed(RNG_SEED)
}

pub struct Creator {
    species: Vec<Species>
}

impl Creator {
    pub fn new() -> Creator {
        Creator {
            species: Vec::new()
        }
    }
    pub fn next_generation(&mut self, genomes: Vec<&Genome>) -> Vec<Genome> {
        info!("Best fitness: {:?}", genomes.iter().fold(genomes[0],
            |a, b| if a.fitness > b.fitness { a } else { b } ));

        // Classify genomes
        for genome in genomes.iter() {
            self.add_genome((*genome).clone());
        }

        // Kill the weak
        for spec in self.species.iter_mut() {
            spec.calc_average_fitness();
            spec.cull();
        }

        // Remove species with no genomes
        self.species.retain(|spec| spec.genomes.len() > 0);

        // Elect representatives
        for spec in self.species.iter_mut() {
            spec.assign_representative();
        }

        // Reproduce based on fitness
        let num_offspring = genomes.len();
        let mut offspring: Vec<Genome> = Vec::with_capacity(num_offspring);
        let total_avg_fitness = self.species.iter().fold(0f64, |acc, s| s.average_fitness() + acc);
        for spec in self.species.iter() {
            let fitness_percentage = spec.average_fitness() / total_avg_fitness;
            let num_spec_offspring = fitness_percentage * num_offspring as f64;
            for _ in 0..(num_spec_offspring as usize) {
                offspring.push(spec.breed_child());
            }
        }

        info!("Next generation has {} species", self.species.len());
        return offspring;
    }

    fn add_genome(&mut self, genome: Genome) {
        for spec in self.species.iter_mut() {
            if spec.compatible(&genome) {
                spec.add_genome(genome);
                return;
            }
        }

        let spec = Species {
            representative: genome.clone(),
            genomes: vec![genome.clone()],
            avg_fitness: 0f64
        };
        self.species.push(spec);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neat::genetics::{Gene, Genome};

    #[test]
    fn creator_maintains_population() {
        let mut genome1 = Genome::new(vec![
            Gene{ into: 0, out: 3, weight: 1.0, enabled: true, innovation: 1 },
            Gene{ into: 1, out: 3, weight: 1.0, enabled: true, innovation: 2 },
            Gene{ into: 3, out: 2, weight: 1.0, enabled: true, innovation: 3 }
        ], 2, 1);

        let mut genome2 = Genome::new(vec![
            Gene{ into: 1, out: 3, weight: 1.0, enabled: true, innovation: 1 },
            Gene{ into: 2, out: 3, weight: 1.0, enabled: true, innovation: 2 },
            Gene{ into: 3, out: 2, weight: 1.0, enabled: true, innovation: 3 }
        ], 2, 1);

        genome1.fitness = -2.0;
        genome2.fitness = -2.0;

        let mut creator = Creator::new();
        let next_gen = creator.next_generation(vec![&genome1, &genome2]);

        assert!(next_gen.len() == 2);
    }
}
