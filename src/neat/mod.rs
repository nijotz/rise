pub mod genetics;
pub mod neurology;
pub mod taxonomy;

use neat::genetics::Genome;
use neat::taxonomy::Species;


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
            let num_spec_offspring =
                (spec.average_fitness() / total_avg_fitness * num_offspring as f64) as usize;
            offspring.push(spec.breed_child());
        }

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
