use rand;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};

use std::collections::HashMap;
use std::f64::consts::E;
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
        write!(f, "Gene: {{ into: {}, out: {}, weight: {} }}",
               self.into, self.out, self.weight)
    }
}
static mut INNOVATION: u64 = 0;

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
    fitness: f64,
    pub network: Network,
    num_inputs: u64,
    num_outputs: u64,
    mutation_rates: MutationRates,
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
        let mut rng = rand::thread_rng();
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
        let mut rng = rand::thread_rng();
        if Range::new(0f64, 1f64).ind_sample(&mut rng) > self.mutation_rates.crossover {
            let child = self.cross(genome);
        }
        let mut child = self.clone();
        child.mutate();
        return child;
    }

    pub fn mutate(&mut self) {
        let mut rng = rand::thread_rng();

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
        let mut rng = rand::thread_rng();
        let num_genes = Range::new(0usize, self.genes.len());
        let weight_step = Range::new(-self.mutation_rates.weight_step, self.mutation_rates.weight_step);
        self.genes[num_genes.ind_sample(&mut rng)].weight *= self.mutation_rates.weight_step;
    }

    pub fn mutate_link(&mut self) {
        let mut rng = rand::thread_rng();
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

        let mut gene = Gene {
            into: neuron1,
            out: neuron2,
            //TODO: make random
            weight: 1.0,
            enabled: true,
            innovation: 0
        };


        self.genes.push(gene);
    }

    pub fn mutate_node(&mut self) {
        if self.genes.len() == 0 { return; }

        let mut rng = rand::thread_rng();
        let gene_range = Range::new(0, (self.genes.len() - 1) as usize);
        let mut gene = self.genes[gene_range.ind_sample(&mut rng)];

        gene.enabled = false;

        let maxneuron = *(self.network.neurons.keys().max().expect("BRAIN DAMAGE"));

        let mut gene1 = gene.clone();
        gene1.out = maxneuron;
        gene1.weight = 1.0;
        unsafe { INNOVATION += 1; gene1.innovation = INNOVATION; }
        gene1.enabled = true;
        self.genes.push(gene1);

        let mut gene2 = gene.clone();
        gene2.into = maxneuron;
        unsafe { INNOVATION += 1; gene2.innovation = INNOVATION; }
        gene2.enabled = true;
        self.genes.push(gene2);
    }

    pub fn cross(&self, genome: &Genome) -> Genome {
        let genome1 = &self;
        let genome2 = genome;
        if self.fitness < genome.fitness {
            let genome2 = &self;
            let genome1 = genome;
        }

        // Build innovations hash to match up genes using historical markings
        let mut innovations: HashMap<u64, Gene> = HashMap::new();
        for gene in genome.genes.iter() {
            innovations.insert(gene.innovation, *gene);
        }

        // Cross genomes
        let mut rng = rand::thread_rng();
        let mut child_genes: Vec<Gene> = Vec::new();
        for gene1 in self.genes.iter() {
            let gene2 = innovations.get(&gene1.innovation).expect("BRAIN DAMAGE");
            let mut gene = gene1.clone();
            if rng.gen() {
                gene = gene2.clone();
            }
            gene.enabled = true;
            if !gene1.enabled || !gene.enabled {
                let zero_to_one = Range::new(0f64, 1f64);
                if zero_to_one.ind_sample(&mut rng) < self.mutation_rates.disable {
                    gene.enabled = false;
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


fn sigmoid(x: f64) -> f64 {
    2.0 / (1.0 + E.powf(-4.9 * x)) - 1.0
}

struct Neuron {
    weights: Vec<f64>,
    incoming: Vec<u64>
}

struct CalcNeuron {
    value: f64,
    calculated: bool
}

impl Neuron {
    fn new() -> Neuron {
        Neuron {
            weights: Vec::new(),
            incoming: Vec::new()
        }
    }

    fn uncalculated_inputs(&self, calc_neurons: &HashMap<u64, CalcNeuron>) -> Vec<u64> {
        let mut clone = self.incoming.clone();
        clone.retain(
            |&n|
            if let Some(x) = calc_neurons.get(&n) {
                debug!("Input #{} calculated: {}", n, x.calculated);
                return !x.calculated;
            } else {
                panic!("BRAIN DAMAGE: Missing neuron");
            }
        );
        debug!("Inputs not yet calculated: {}", clone.len());
        clone
    }
}

pub struct Network {
    neurons: HashMap<u64, Neuron>,
    num_inputs: u64,
    num_outputs: u64
}

impl Network {
    pub fn new(genes: &Vec<Gene>, num_inputs: u64, num_outputs: u64) -> Network {
        let mut neurons: HashMap<u64, Neuron> = HashMap::new();

        // Add inputs and outputs to network
        for i in 0..num_inputs {
            debug!("Creating input neuron #{}", i);
            neurons.insert(i, Neuron::new());
        }

        for i in num_inputs..(num_inputs + num_outputs) {
            debug!("Creating output neuron #{}", i);
            neurons.insert(i, Neuron::new());
        }

        // Use genes to build hidden layer
        for gene in genes.iter() {
            debug!("Processing gene: {:?}", gene);
            if !gene.enabled { continue; }

            if !neurons.contains_key(&gene.out) {
                debug!("Creating neuron #{}", gene.out);
                neurons.insert(gene.out, Neuron::new());
            }

            match neurons.get_mut(&gene.out) {
                Some(neuron) => {
                    debug!("Modifying neuron #{} - incoming: {}",
                           gene.out, gene.into);
                    neuron.incoming.push(gene.into);

                    debug!("Modifying neuron #{} - weight: {}",
                           gene.out, gene.weight);
                    neuron.weights.push(gene.weight);
                },
                _ => { panic!("BRAIN DAMAGE: Missing neuron") }
            }

            if !neurons.contains_key(&gene.into) {
                debug!("Creating neuron #{}", gene.into);
                neurons.insert(gene.into, Neuron::new());
            }
        }

        return Network {
            neurons: neurons,
            num_inputs: num_inputs,
            num_outputs: num_outputs
        };
    }

    pub fn evaluate(&self, inputs: Vec<f64>) -> Vec<f64> {
        // Initialize a network of neuron calculations
        let size = self.neurons.len();
        let mut calc_neurons: HashMap<u64, CalcNeuron> = HashMap::with_capacity(size);
        for i in self.neurons.keys() {
            calc_neurons.insert(*i, CalcNeuron{ value: 0f64, calculated: false });
        }

        // Feed inputs into input layer
        for (i, &input) in inputs.iter().enumerate() {
            if let Some(input_calc) = calc_neurons.get_mut(&(i as u64)) {
                input_calc.value = input;
                input_calc.calculated = true;
            }
        }

        // Initialize traversal data with output nodes
        let mut need: Vec<u64> = Vec::new();
        for x in self.num_inputs..(self.num_inputs + self.num_outputs) {
            need.push(x);
        }

        // Traverse network for outputs inward
        while let Some(node) = need.pop() {
            debug!("Evaluating neuron #{}", node);
            if let Some(neuron) = self.neurons.get(&node) {
                if neuron.uncalculated_inputs(&calc_neurons).len() == 0 {
                    debug!("Neuron #{} has necessary inputs", node);
                    debug!("Neuron #{} weights: {:?}", node, neuron.weights);
                    debug!("Neuron #{} incoming: {:?}", node, neuron.incoming);
                    let zip_iter = neuron.incoming.iter().zip(neuron.weights.iter());
                    let sum = zip_iter.fold(0f64, |sum, (&incoming, &weight)|
                        if let Some(in_neuron) = calc_neurons.get_mut(&incoming) {
                            debug!("Neuron #{} applies weight {} to value {} from neuron #{}",
                                   node, weight, in_neuron.value, incoming);
                            return sum + (in_neuron.value * weight);
                        } else {
                            return sum;
                        }
                    );

                    if let Some(calc) = calc_neurons.get_mut(&node) {
                        debug!("Applying sigmoid to sum {} for neuron #{}",
                               sum, node);
                        calc.value = sigmoid(sum);
                        calc.calculated = true;
                    }
                } else {
                    // Re-calculate this one after its inputs have been calculated by pushing it on
                    // the stack before inputs are pushed on.
                    need.push(node);

                    let in_needs = neuron.uncalculated_inputs(&calc_neurons);
                    debug!("Neuron #{} needs inputs: {:?}", node, in_needs);
                    let mut pushed = false;

                    for x in in_needs {
                        // If the inputs we need are already on the stack, don't visit them again.
                        // Prevents circular dependencies causing an infinite loop.
                        if !need.iter().any(|i| x == *i) {
                            need.push(x);
                            pushed = true;
                        } else {
                            debug!("Neuron #{} already needed, not pushing again", x);
                        }

                    }

                    // If dependencies weren't pushed on, don't push this one on and set it to
                    // calculated to prevent if from being pushed on again.
                    if !pushed {
                        if let Some(calc_neuron) = calc_neurons.get_mut(&node) {
                            calc_neuron.calculated = true;
                        }
                        need.pop();
                    }
                }
            }
        }

        // Get output values
        let mut outputs: Vec<f64> = Vec::new();
        for i in 0..self.num_outputs {
            outputs.push(calc_neurons[&(i + self.num_inputs)].value);
        }

        debug!("Calculated output: {:?}", outputs);

        return outputs;
    }
}
