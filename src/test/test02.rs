use anyhow::Result;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device};

const IMAGE_DIM:i64 = 784;
const HIDDEN_NODES:i64 = 128;
const LABELS:i64 = 10;

fn net(vs: &nn::Path) -> impl Module{
    nn::seq()
        .add(nn::linear(
            vs / "layer1",
            IMAGE_DIM,
            HIDDEN_NODES,
            Default::default()
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(
            vs,
            HIDDEN_NODES,
            LABELS,
            Default::default()
        ))
}

#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn test() -> Result<()>{
        // let m = tch::vision::mnist::load_dir("src/test")?;
        // println!("{:?}", m);

        Ok(())
    }
}