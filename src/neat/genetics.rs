use neat;
use neat::neurology::Network;

use rand::Rng;
use rand::distributions::{IndependentSample, Range};

use std::collections::HashMap;
use std::fmt;

#[derive(Copy, Clone)]
pub struct Gene {
    pub into: u64,
    pub out: u64,
    pub weight: f64,
    pub enabled: bool,
    pub innovation: u64
}

impl fmt::Debug for Gene {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{ I:{:02}, {} => {}, W:{:.2} ON:{} }}",
               self.innovation, self.into, self.out, self.weight, self.enabled)
    }
}

static mut INNOVATION: u64 = 0;

unsafe fn innovation_next() -> u64 {
    INNOVATION += 1;
    return INNOVATION;
}

const MUTATE_CROSSOVER: f64 = 0.75;
const MUTATE_WEIGHT: f64 = 0.05;
const MUTATE_LINK: f64 = 0.05;
const MUTATE_NODE: f64 = 0.05;
const MUTATE_WEIGHT_STEP: f64 = 0.05;
const MUTATE_DISABLE: f64 = 0.8;

#[derive(Copy, Clone)]
pub struct MutationRates {
    crossover: f64,
    weight: f64,
    weight_step: f64,
    link: f64,
    node: f64,
    disable: f64
}

impl MutationRates {
    pub fn new() -> MutationRates {
        MutationRates {
            crossover: MUTATE_CROSSOVER,
            weight: MUTATE_WEIGHT,
            weight_step: MUTATE_WEIGHT_STEP,
            link: MUTATE_LINK,
            node: MUTATE_NODE,
            disable: MUTATE_DISABLE
        }
    }
}

pub struct Genome {
    pub genes: Vec<Gene>,
    pub fitness: f64,
    pub network: Network,
    num_inputs: u64,
    num_outputs: u64,
    mutation_rates: MutationRates,
}

impl fmt::Debug for Genome {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Genome: {:?}, Fitness: {}",
               self.genes, self.fitness)
    }
}

impl Genome {
    pub fn new(genes: Vec<Gene>, num_inputs: u64, num_outputs: u64) -> Genome {
        let network = Network::new(&genes, num_inputs, num_outputs);
        Genome {
            genes: genes,
            network: network,
            fitness: 0.0,
            num_inputs: num_inputs,
            num_outputs: num_outputs,
            mutation_rates: MutationRates::new()
        }
    }

    pub fn clone(&self) -> Genome {
        let clone_genes = (*self).genes.clone();
        let clone_network = Network::new(&clone_genes, self.num_inputs, self.num_outputs);
        let clone = Genome {
            genes: clone_genes,
            fitness: self.fitness,
            network: clone_network,
            num_inputs: self.num_inputs,
            num_outputs: self.num_outputs,
            mutation_rates: self.mutation_rates
        };
        return clone;
    }

    pub fn random(num_inputs: u64, num_outputs: u64) -> Genome {
        let mut genes = Vec::new();
        let num_genes = Range::new(1u64, 5u64);
        let num_neurons = Range::new(1u64, 7u64);
        let weights = Range::new(-1f64, 1f64);
        let mut rng = neat::rng();
        for i in 0..num_genes.ind_sample(&mut rng) {
            let gene = Gene {
                into: num_neurons.ind_sample(&mut rng),
                out: num_neurons.ind_sample(&mut rng),
                weight: weights.ind_sample(&mut rng),
                enabled: true,
                innovation: i
            };
            genes.push(gene);
        }

        return Genome::new(genes, num_inputs, num_outputs);
    }

    pub fn breed(&self, genome: &Genome) -> Genome {
        debug!("Breeding genomes");
        let mut child = self.clone();

        let mut rng = neat::rng();
        if Range::new(0f64, 1f64).ind_sample(&mut rng) > self.mutation_rates.crossover {
            debug!("Crossing genomes");
            child = self.cross(genome);
        }

        child.mutate();
        return child;
    }

    pub fn mutate(&mut self) {
        let mut rng = neat::rng();

        let zero_to_one = Range::new(0f64, 1f64);
        if zero_to_one.ind_sample(&mut rng) < self.mutation_rates.weight {
            self.mutate_weight();
        }

        let zero_to_one = Range::new(0f64, 1f64);
        if zero_to_one.ind_sample(&mut rng) < self.mutation_rates.link {
            self.mutate_link();
        }

        let zero_to_one = Range::new(0f64, 1f64);
        if zero_to_one.ind_sample(&mut rng) < self.mutation_rates.node {
            self.mutate_node();
        }
    }

    pub fn mutate_weight(&mut self) {
        let mut rng = neat::rng();
        let num_genes = Range::new(0usize, self.genes.len());
        let weight_step_range = Range::new(
            -self.mutation_rates.weight_step,
            self.mutation_rates.weight_step);
        let gene = num_genes.ind_sample(&mut rng);
        let weight_step = weight_step_range.ind_sample(&mut rng) + 1f64;

        debug!("Mutating weight of gene #{} by {:.2}%", gene, weight_step);
        self.genes[gene].weight *= weight_step;
    }

    pub fn mutate_link(&mut self) {
        let mut rng = neat::rng();
        let neuron_range = Range::new(0u64, self.network.neurons.keys().len() as u64);
        let mut neuron1 = neuron_range.ind_sample(&mut rng);
        let mut neuron2 = neuron_range.ind_sample(&mut rng);

        // Both input nodes
        if neuron1 <= self.num_inputs - 1 && neuron2 <= self.num_inputs - 1 {
            return;
        }

        // Only neuron2 is input so swap direction
        if neuron2 <= self.num_inputs - 1 {
            let temp = neuron2;
            neuron2 = neuron1;
            neuron1 = temp;
        }

        // Check for existing neuron link
        for gene in self.genes.iter() {
            if gene.into == neuron1 && gene.out == neuron2 {
                return;
            }
        }

        let gene = Gene {
            into: neuron1,
            out: neuron2,
            //TODO: make random
            weight: 1.0,
            enabled: true,
            innovation: 0
        };

        debug!("Mutating new link: {} -> {}", neuron1, neuron2);
        self.genes.push(gene);
    }

    pub fn mutate_node(&mut self) {
        if self.genes.len() == 0 { return; }

        let mut rng = neat::rng();
        let gene_range = Range::new(0, self.genes.len());
        let mut gene = self.genes[gene_range.ind_sample(&mut rng)];

        gene.enabled = false;

        let maxneuron = *(self.network.neurons.keys().max().expect("BRAIN DAMAGE"));
        debug!("Mutating new node between {} -> {}", gene.into, gene.out);

        let mut gene1 = gene.clone();
        gene1.out = maxneuron;
        gene1.weight = 1.0;
        unsafe { gene1.innovation = innovation_next(); }
        gene1.enabled = true;
        self.genes.push(gene1);

        let mut gene2 = gene.clone();
        gene2.into = maxneuron;
        unsafe { gene2.innovation = innovation_next(); }
        gene2.enabled = true;
        self.genes.push(gene2);
    }

    pub fn cross(&self, genome: &Genome) -> Genome {
        let mut genome1 = self;
        let mut genome2 = genome;
        if self.fitness < genome.fitness {
            genome2 = self;
            genome1 = genome;
        }

        // Build innovations hash to match up genes using historical markings
        let mut innovations2: HashMap<u64, Gene> = HashMap::new();
        for gene in genome2.genes.iter() {
            innovations2.insert(gene.innovation, *gene);
        }

        // Cross genomes
        let mut rng = neat::rng();
        let mut child_genes: Vec<Gene> = Vec::new();
        for gene1 in genome1.genes.iter() {
            let mut gene = gene1.clone();
            gene.enabled = true;
            if let Some(gene2) = innovations2.get(&gene1.innovation) {
                if rng.gen() {
                    gene = gene2.clone();
                }
                if !gene1.enabled || !gene2.enabled {
                    let zero_to_one = Range::new(0f64, 1f64);
                    if zero_to_one.ind_sample(&mut rng) < self.mutation_rates.disable {
                        gene.enabled = false;
                    }
                }
            }
            child_genes.push(gene);
        }

        let network = Network::new(&child_genes, genome1.num_inputs, genome1.num_outputs);
        let child = Genome {
            genes: child_genes,
            fitness: 0f64,
            network: network,
            num_inputs: genome1.num_inputs,
            num_outputs: genome1.num_outputs,
            mutation_rates: genome1.mutation_rates
        };

        return child;
    }
}

#[cfg(test)]
mod tests {
    extern crate env_logger;

    use super::*;

    #[test]
    fn genomes_breed() {
        let genome1 = Genome::new(vec![
            Gene{ into: 0, out: 3, weight: 1.0, enabled: true, innovation: 1 },
            Gene{ into: 1, out: 3, weight: 1.0, enabled: true, innovation: 2 },
            Gene{ into: 3, out: 2, weight: 1.0, enabled: true, innovation: 3 }
        ], 2, 1);

        let genome2 = Genome::new(vec![
            Gene{ into: 1, out: 3, weight: 1.0, enabled: true, innovation: 1 },
            Gene{ into: 2, out: 3, weight: 1.0, enabled: true, innovation: 2 },
            Gene{ into: 3, out: 2, weight: 1.0, enabled: true, innovation: 3 }
        ], 2, 1);

        let child = genome1.breed(&genome2);

        assert!(child.genes.len() > 0);
    }

    #[test]
    fn genome_crossover_preserves_innovation_ordering() {
        let _ = env_logger::init();
        unsafe { super::INNOVATION = 6 }
        let genome1 = Genome::new(vec![
            Gene{ into: 0, out: 3, weight: 1.0, enabled: true, innovation: 1 },
            Gene{ into: 1, out: 3, weight: 1.0, enabled: true, innovation: 2 },
            Gene{ into: 3, out: 2, weight: 1.0, enabled: true, innovation: 3 }
        ], 2, 1);

        let genome2 = Genome::new(vec![
            Gene{ into: 1, out: 3, weight: 1.0, enabled: true, innovation: 4 },
            Gene{ into: 2, out: 3, weight: 1.0, enabled: true, innovation: 5 },
            Gene{ into: 3, out: 2, weight: 1.0, enabled: true, innovation: 6 }
        ], 2, 1);

        let child = genome1.cross(&genome2);

        let mut ordered = true;
        let mut innovation: u64 = 0;
        for gene in child.genes.iter() {
            if innovation >= gene.innovation {
                ordered = false;
                break;
            }

            innovation = gene.innovation;
        }
        assert!(ordered);
    }
}
