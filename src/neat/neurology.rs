use neat::genetics::Gene;

use std::f64::consts::E;
use std::collections::HashMap;

fn sigmoid(x: f64) -> f64 {
    2.0 / (1.0 + E.powf(-4.9 * x)) - 1.0
}

pub struct Neuron {
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
    pub neurons: HashMap<u64, Neuron>,
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

#[cfg(test)]
mod tests {
    use neat::genetics::{Gene, Genome};

    #[test]
    fn networks_evaluate() {
        let genome = Genome::new(vec![
            Gene{ into: 0, out: 3, weight: 1.0, enabled: true, innovation: 1 },
            Gene{ into: 1, out: 3, weight: 1.0, enabled: true, innovation: 2 },
            Gene{ into: 3, out: 2, weight: 1.0, enabled: true, innovation: 3 }
        ], 2, 1);

        let inputs = vec![1f64, 1f64];
        let outputs = genome.network.evaluate(inputs);

        assert!(outputs.len() == 1);
        assert!(outputs[0] >= 0f64 && outputs[0] <= 1f64);
    }

    #[test]
    fn networks_handles_circular_dependencies() {
        let genome = Genome::new(vec![
            Gene{ into: 1, out: 3, weight: 1.0, enabled: true, innovation: 1 },
            Gene{ into: 2, out: 3, weight: 1.0, enabled: true, innovation: 2 },
            Gene{ into: 3, out: 2, weight: 1.0, enabled: true, innovation: 3 }
        ], 2, 1);

        let inputs = vec![1f64, 1f64];
        let outputs = genome.network.evaluate(inputs);

        assert!(outputs.len() == 1);
        assert!(outputs[0] >= 0f64 && outputs[0] <= 1f64);
    }
}
