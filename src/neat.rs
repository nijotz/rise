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
    adjustedFitness: f64,
    network: Vec<Neuron>,
    mexneuron: usize,
    globalRank: u64,
    mutationRates: MutationRates
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
