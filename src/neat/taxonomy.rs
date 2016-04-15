use neat::genetics::Genome;

const EXCESS_COEFF: f64 = 1.0;
const DISJOINT_COEFF: f64 = 1.0;
const WEIGHT_COEFF: f64 = 1.0;
const DIFFERENCE_THRESHOLD: f64 = 1.0;

pub struct Species {
    pub representative: Genome,
    pub genomes: Vec<Genome>
}

impl Species {
    pub fn difference(genome1: &Genome, genome2: &Genome) -> f64 {
        let mut disjoint = Vec::new();
        let mut excess = Vec::new();
        let mut weight_diffs: Vec<f64> = Vec::new();
        let mut g1 = 0;
        let mut g2 = 0;

        while g1 < genome1.genes.len() || g2 < genome2.genes.len() {
            let gene1 = genome1.genes[g1];
            let gene2 = genome2.genes[g2];

            if gene1.innovation == gene2.innovation {
                weight_diffs.push( (gene1.weight - gene2.weight).abs() );
                g1 += 1;
                g2 += 1;
            }

            if gene1.innovation > gene2.innovation {
                if g1 == genome1.genes.len() - 1 {
                    excess.push(gene2.innovation)
                } else {
                    disjoint.push(gene2.innovation);
                }
                g2 += 1;
            }

            if gene2.innovation > gene1.innovation {
                if g2 == genome2.genes.len() - 1 {
                    excess.push(gene1.innovation)
                } else {
                    disjoint.push(gene1.innovation);
                }
                g1 += 1;
            }
        }

        let weight_diff =
            weight_diffs.iter().fold(0f64, |acc, x| acc + x) /
            weight_diffs.len() as f64;

        let mut gene_size: f64 = genome1.genes.len() as f64;
        if genome2.genes.len() > genome1.genes.len() {
            gene_size = genome2.genes.len() as f64;
        }

        return ( (EXCESS_COEFF * excess.len() as f64) / gene_size) +
            ( (DISJOINT_COEFF * disjoint.len() as f64) / gene_size) +
            ( WEIGHT_COEFF * weight_diff );
    }

    pub fn compatible(&self, genome: Genome) -> bool {
        Species::difference(&self.representative, &genome) > DIFFERENCE_THRESHOLD
    }
}
