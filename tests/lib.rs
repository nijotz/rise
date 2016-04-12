#[macro_use]
extern crate log;


#[cfg(test)]
mod tests {
    extern crate env_logger;

    extern crate rise;
    use super::*;
    use self::rise::neat::{Gene, Genome};

    #[test]
    fn it_works() {
        let _ = env_logger::init();
        let genome = Genome::new(vec![
            Gene{ into: 0, out: 3, weight: 1.0, enabled: true },
            Gene{ into: 1, out: 3, weight: 1.0, enabled: true },
            Gene{ into: 3, out: 2, weight: 1.0, enabled: true }
        ], 2, 1);

        let inputs = vec![1f64, 1f64];
        let outputs = genome.network.evaluate(inputs);

        assert!(outputs.len() == 1);
        assert!(outputs[0] >= 0f64 && outputs[0] <= 1f64);
    }

    #[test]
    fn it_handles_circular_dependencies() {
        let _ = env_logger::init();
        let genome = Genome::new(vec![
            Gene{ into: 1, out: 3, weight: 1.0, enabled: true },
            Gene{ into: 2, out: 3, weight: 1.0, enabled: true },
            Gene{ into: 3, out: 2, weight: 1.0, enabled: true }
        ], 2, 1);

        let inputs = vec![1f64, 1f64];
        let outputs = genome.network.evaluate(inputs);

        assert!(outputs.len() == 1);
        assert!(outputs[0] >= 0f64 && outputs[0] <= 1f64);
    }
}
