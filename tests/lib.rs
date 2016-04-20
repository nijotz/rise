#[macro_use]
extern crate log;


#[cfg(test)]
mod tests {
    extern crate env_logger;

    extern crate rise;
    use self::rise::neat::genetics::{Gene, Genome};

    #[test]
    fn it_works() {
        let _ = env_logger::init();
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
    fn it_handles_circular_dependencies() {
        let _ = env_logger::init();
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

    #[test]
    fn it_breeds() {
        let _ = env_logger::init();

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

    use self::rise::neat::taxonomy::Species;

    fn it_can_measure_compatibility() {
        let _ = env_logger::init();

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

        let species = Species {
            representative: genome1,
            genomes: Vec::new(),
            avg_fitness: 0f64
        };

        assert!(species.compatible(&genome2) == true);
    }
}
