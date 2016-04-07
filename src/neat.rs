use std::collections::HashMap;
use std::f64::consts::E;

pub struct Gene {
    into: u64,
    out: u64,
    weight: f64,
    enabled: bool,
    //TODO: historical_marker: u64
}

pub struct Genome {
    genes: Vec<Gene>,
    fitness: f64,
    network: Network,
    num_inputs: u64,
    num_outputs: u64,
}

impl Genome {
    pub fn new(genes: Vec<Gene>, num_inputs: u64, num_outputs: u64) -> Genome {
        let network = Network::new(&genes, num_inputs, num_outputs);
        return Genome {
            genes: genes,
            network: network,
            fitness: 0.0,
            num_inputs: num_inputs,
            num_outputs: num_outputs,
        }
    }
}


fn sigmoid(x: f64) -> f64 {
    2.0 / (1.0 + E.powf(-4.9 * x)) - 1.0
}

pub struct Neuron {
    weight: f64,
    incoming: Vec<u64>
}

pub struct CalcNeuron {
    value: f64,
    calculated: bool
}

impl Neuron {
    fn new() -> Neuron {
        Neuron {
            weight: 0f64,
            incoming: Vec::new()
        }
    }

    fn calculated_inputs(&self, calc_neurons: &HashMap<u64, CalcNeuron>) -> Vec<u64> {
        let mut clone = self.incoming.clone();
        clone.retain(
            |&n|
            if let Some(x) = calc_neurons.get(&n) {
                return x.calculated;
            } else {
                panic!("BRAIN DAMAGE: Missing neuron");
            }
        );
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
            neurons.insert(i, Neuron::new());
        }

        for i in num_inputs..(num_inputs + num_outputs) {
            neurons.insert(i, Neuron::new());
        }

        // Use genes to build hidden layer
        for gene in genes {
            if !gene.enabled { continue; }
            if !neurons.contains_key(&gene.out) {
                neurons.insert(gene.out, Neuron::new());
            }

            match neurons.get_mut(&gene.out) {
                Some(neuron) => neuron.incoming.push(gene.into),
                _ => { panic!("BRAIN DAMAGE: Missing neuron") }
            }

            if !neurons.contains_key(&gene.into) {
                neurons.insert(gene.into, Neuron::new());
            }
        }

        return Network { neurons: neurons, num_inputs: num_inputs, num_outputs: num_outputs };
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
            if let Some(x) = calc_neurons.get_mut(&(i as u64)) {
                x.value = input;
            }
        }


        // Initialize traversal data with output nodes
        let mut need: Vec<u64> = Vec::new();
        for x in self.num_inputs..(self.num_inputs + self.num_outputs) {
            need.push(x);
        }

        // Traverse network for outputs inward
        while let Some(node) = need.pop() {
            if let Some(neuron) = self.neurons.get(&node) {
                if neuron.calculated_inputs(&calc_neurons).len() == 0 {
                    let sum = neuron.incoming.iter().fold(0f64, |sum, x| if let Some(y) = calc_neurons.get_mut(x) { return sum + y.value } else { return sum; }) * neuron.weight;
                    if let Some(x) = calc_neurons.get_mut(&node) {
                        x.value = sigmoid(sum);
                        x.calculated = true;
                    }
                } else {
                    need.push(node);
                    for x in neuron.calculated_inputs(&calc_neurons) {
                        // prevents circular dependencies causing an infinite loop
                        if !need.iter().any(|i| x == *i) {
                            need.push(x);
                        }
                    }
                }
            }
        }

        // Get output values
        let mut outputs: Vec<f64> = Vec::new();
        for i in 0..self.num_outputs {
            outputs[i as usize] = calc_neurons[&(i + self.num_inputs)].value;
        }

        return outputs;
    }
}
