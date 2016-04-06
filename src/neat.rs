const MUTATE_CONNECTIONS_CHANCE: f64 = 0.05;
const MUTATE_LINK_CHANCE: f64 = 0.05;
const MUTATE_BIAS_CHANCE: f64 = 0.05;
const MUTATE_NODE_CHANCE: f64 = 0.05;
const MUTATE_ENABLE_CHANCE: f64 = 0.05;
const MUTATE_DISABLE_CHANCE: f64 = 0.05;
const MUTATE_STEP_CHANCE: f64 = 0.05;

pub struct Gene {
    into: u64,
    out: u64,
    weight: f64,
    enabled: bool,
    innovation: u64
}

pub struct Genome {
    genes: Vec<Gene>,
    fitness: f64,
    adjusted_fitness: f64,
    network: Vec<Neuron>,
    inputs: usize,
    global_rank: u64,
    mutation_rates: MutationRates
}

pub struct MutationRates {
    connections: f64,
    link: f64,
    bias: f64,
    node: f64,
    enable: f64,
    disable: f64,
    step: f64
}

pub struct Neuron {
    value: f64,
    incoming: Vec<usize>
}

impl Genome {
    pub fn new() -> Genome {
        Genome {
            genes: Vec::new(),
            network: Vec::new(),
            fitness: 0.0,
            adjusted_fitness: 0.0,
            inputs: 0,
            global_rank: 0,
            mutation_rates: MutationRates::new()
        }
    }
}

impl MutationRates {
    pub fn new() -> MutationRates {
        MutationRates {
            connections: MUTATE_CONNECTIONS_CHANCE,
            link: MUTATE_LINK_CHANCE,
            bias: MUTATE_BIAS_CHANCE,
            node: MUTATE_NODE_CHANCE,
            enable: MUTATE_ENABLE_CHANCE,
            disable: MUTATE_DISABLE_CHANCE,
            step: MUTATE_STEP_CHANCE
        }
    }
}
